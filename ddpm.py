import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import math
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2
timestep_embed_dim = 128
hidden_dims = [256, 256]
n_timesteps = 500
beta_range = [1e-4, 0.02]
batch_size = 256
lr = 4e-4
epochs = 1000
seed = 2024

torch.manual_seed(seed)
np.random.seed(seed)

class GMMDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.FloatTensor(np.load(file_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Sinusoidal embedding for diffusion timestep (Attention is All You Need)
# 一种位置编码，前一半sin后一半cos
# https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Denoiser model
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, hidden_dims[0]),
            nn.SiLU(), # SiLU activation function: SiLU(x) = x * sigmoid(x) = x * frac{1}{1 + exp(-x)}
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[1], input_dim)
        )
        
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = self.input_proj(x) + t_emb
        return self.mlp(x)
    
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Denoiser().to(device)
        self.n_steps = n_timesteps
        
        self.betas = torch.linspace(beta_range[0], beta_range[1], n_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # alpha_t = \prod (1 - beta_i), here is alpha_bar_t

    # def extract(v, t, x_shape):
    #     # v[T]
    #     # t[B] x_shape = [B,C,H,W]
    #     out = torch.gather(v, index=t, dim=0).float()
    #     # [B,1,1,1],分别代表batch_size,通道数,长,宽
    #     return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    def extract(self, a, t, x_shape):
        '''
        辅助函数，用于根据时间步t从一个序列中提取对应的值，并调整形状以匹配输入数据x的形状。
        '''
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,)*(len(x_shape)-1)))
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            
        alpha_bar = self.extract(self.alpha_bars, t, x0.shape)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    def p_losses(self, x0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,)).to(device)
        noise = torch.randn_like(x0)
        
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_noisy, t)
        
        return F.mse_loss(pred_noise, noise)
    
    def sample(self, n_samples):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, input_dim).to(device)
            
            for t in reversed(range(self.n_steps)):
                t_batch = torch.full((n_samples,), t).to(device)
                pred_noise = self.model(x, t_batch)
                
                alpha = self.extract(self.alphas, t_batch, x.shape)
                alpha_bar = self.extract(self.alpha_bars, t_batch, x.shape)
                
                z = torch.randn_like(x) if t > 0 else 0

                x = (x - (1 - alpha)/torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
                x += torch.sqrt(self.betas[t]) * z
                
        return x

def train():
    train_set = GMMDataset('./data/train.npy')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    model = Diffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = model.p_losses(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'ddpm_best_model.pth')

# 评估函数
def evaluate(model_path):
    model = Diffusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 生成样本
    samples = model.sample(1600).cpu().numpy()
    
    # 加载验证数据
    val_set = GMMDataset('./data/val.npy')
    val_data = val_set.data.numpy()
    val_data = val_data[:1600]

    plt.figure(figsize=(10,10))
    plt.scatter(samples[:,0], samples[:,1], s=5, c='r', label = 'Generated')
    plt.scatter(val_data[:,0], val_data[:,1], s=5, c='b', label = 'Validation')
    plt.legend()
    plt.title("DDPM Generated vs Validation Samples")
    plt.savefig('ddpm_generated_vs_validation_samples.png')

if __name__ == "__main__":
    train()
    evaluate('ddpm_best_model.pth')
