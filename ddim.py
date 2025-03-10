import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
import logging
import math
from tqdm import tqdm
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2
timestep_embed_dim = 128
hidden_dims = [256, 256]
n_timesteps = 1000
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

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, hidden_dims[0]),
            nn.SiLU(),
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

    def extract(self, a, t, x_shape):
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
                t_batch = torch.full((n_samples,), t, device=device)
                pred_noise = self.model(x, t_batch)

                alpha = self.extract(self.alphas, t_batch, x.shape)
                alpha_bar = self.extract(self.alpha_bars, t_batch, x.shape)
                
                z = torch.randn_like(x) if t > 0 else 0
                x = (x - (1 - alpha)/torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
                x += torch.sqrt(self.betas[t]) * z
        return x

    def sample_ddim(self, n_samples, num_steps=50, eta=0.0):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, input_dim).to(device)
            times = torch.linspace(0, self.n_steps-1, steps=num_steps, dtype=torch.long)
            times = torch.unique(times).flip(0).tolist()
            
            for i in range(len(times)-1):
                t = times[i]
                t_prev = times[i+1] if (i+1) < len(times) else 0
                
                t_batch = torch.full((n_samples,), t, device=device)
                pred_noise = self.model(x, t_batch)
                
                alpha_bar_t = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars[t_prev] if t_prev >=0 else torch.tensor(1.0, device=device)
                
                sigma_t = eta * torch.sqrt( (1 - alpha_bar_prev)/(1 - alpha_bar_t) * (1 - alpha_bar_t/alpha_bar_prev) )
                # sigma_t = eta * torch.sqrt(1 - alpha_bar_prev - sigma_t**2) 
                part1 = torch.sqrt(alpha_bar_prev) * (x - torch.sqrt(1 - alpha_bar_t)*pred_noise) / torch.sqrt(alpha_bar_t)
                part2 = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise
                part3 = sigma_t * torch.randn_like(x) if (eta > 0 and t_prev > 0) else 0
                
                x = part1 + part2 + part3
            return x

def train():
    train_set = GMMDataset('./data/train.npy')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Diffusion().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
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
            torch.save(model.state_dict(), 'ddim_best_model.pth')

def evaluate(model_path):
    model = Diffusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    samples = model.sample_ddim(1600, num_steps=50, eta=0).cpu().numpy()
    
    val_set = GMMDataset('./data/val.npy')
    val_data = val_set.data.numpy()[:1600]

    plt.figure(figsize=(10,10))
    plt.scatter(samples[:,0], samples[:,1], s=5, c='r', label='Generated (DDIM)')
    plt.scatter(val_data[:,0], val_data[:,1], s=5, c='b', label='Validation')
    plt.legend()
    plt.title("DDIM Generated vs Validation Samples")
    plt.savefig('ddim_generated_vs_validation.png')
    plt.close()

if __name__ == "__main__":
    train()
    evaluate('ddim_best_model.pth')