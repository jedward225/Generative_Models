# gan.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 2
latent_dim = 2
hidden_dims = [128, 256]
batch_size = 250
lr = 4e-4
epochs = 3600
seed = 2024

torch.manual_seed(seed)
np.random.seed(seed)

class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.data = torch.FloatTensor(np.load(file_path))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], input_dim)
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class GAN:
    def __init__(self):
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train(self, dataloader):
        self.generator.train()
        self.discriminator.train()
        '''
        1. 固定G，求损失最大的D，求JS散度，找到分布差异的度量。
        2. 固定D，最小化第一步的结果，最小化JS散度。
        '''
        for epoch in range(epochs):
            total_loss_G = 0.0
            total_loss_D = 0.0
            
            for real_data in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                real_data = real_data.to(device)
                batch_size = real_data.size(0)
                
                self.optimizer_D.zero_grad()
                
                # 真实数据
                real_labels = torch.ones(batch_size, 1).to(device)
                real_output = self.discriminator(real_data)
                loss_real = self.criterion(real_output, real_labels)
                
                # 生成数据
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_data = self.generator(z)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                fake_output = self.discriminator(fake_data.detach())
                loss_fake = self.criterion(fake_output, fake_labels)
                
                # 总损失和反向传播
                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()
                total_loss_D += loss_D.item()
                
                # 训练生成器
                self.optimizer_G.zero_grad()
                validity = self.discriminator(fake_data)
                loss_G = self.criterion(validity, real_labels)
                loss_G.backward()
                self.optimizer_G.step()
                total_loss_G += loss_G.item()
            
            avg_loss_G = total_loss_G / len(dataloader)
            avg_loss_D = total_loss_D / len(dataloader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss_D: {avg_loss_D:.4f} Loss_G: {avg_loss_G:.4f}")
            
            if (epoch+1) % 100 == 0:
                self.generate_samples(epoch+1)
                torch.save(self.generator.state_dict(), f'gan_generator_epoch{epoch+1}.pth')

    def generate_samples(self, epoch, num_samples=1600):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim).to(device)
            samples = self.generator(z).cpu().numpy()
            
            plt.figure(figsize=(10,10))
            plt.scatter(samples[:,0], samples[:,1], s=5, c='r')
            plt.title(f"Generated Samples at Epoch {epoch}")
            plt.savefig(f'gan_samples_epoch{epoch}.png')
            plt.close()

def evaluate(model_path, num = None):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, weights_only=True))
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(1600, latent_dim).to(device)
        samples = generator(z).cpu().numpy()
    
    val_set = GMMDataset('./data/val.npy')
    val_data = val_set.data.numpy()[:1600]

    plt.figure(figsize=(10,10))
    plt.scatter(samples[:,0], samples[:,1], s=5, c='r', label='Generated (GAN)')
    plt.scatter(val_data[:,0], val_data[:,1], s=5, c='b', label='Validation')
    plt.legend()
    plt.title("GAN Generated vs Validation Samples")
    plt.savefig('gan_generated_vs_validation.png')
    plt.close()

if __name__ == "__main__":
    # train_set = GMMDataset('./data/train.npy')
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # gan = GAN()
    # gan.train(train_loader)
    
    # evaluation
    evaluate('gan_generator_epoch3500.pth')