from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import os
from unet import UNetModel

    
def beta_schedules(beta_start: float = 1e-4, beta_end: float = 2e-2, T: int = 1000) -> Dict[str, torch.Tensor]:

    assert beta_start < beta_end < 1.0, "beta1 and beta2 must be in (0, 1)" 

    b_t = torch.linspace(beta_start, beta_end, T+1, dtype=torch.float32)
    a_t = 1 - b_t
    ab_t = torch.cumsum(torch.log(a_t), dim=0).exp()

    return {
        "b_t": b_t, # \beta_t
        "a_t": a_t,  # \alpha_t
        "ab_t": ab_t,  # \bar{\alpha_t}
    }

class DiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DiffusionModel, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in beta_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        t = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = self.ab_t.sqrt()[t, None, None, None] * x + (1 - self.ab_t[t, None, None, None]).sqrt() * eps # x_t = sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps

        return self.criterion(eps, self.eps_model(x_t, t / self.n_T))

    @torch.inference_mode()
    def ddpm_sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            
            x_i = 1 / self.a_t[i].sqrt() * (x_i - (1 - self.a_t[i]) / (1 - self.ab_t[i]).sqrt() * eps) + (1 - self.a_t[i]).sqrt() * z

        return x_i

    @torch.inference_mode()
    def ddim_sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        for i in range(self.n_T, 0, -1):
            eps = self.eps_model(x_i, i / self.n_T)
            
            pred_x0 = self.ab_t[i-1].sqrt() / self.ab_t[i].sqrt() * (x_i - (1 - self.ab_t[i]).sqrt() * eps)
            dir_xt = (1 - self.ab_t[i-1]).sqrt() * eps

            x_i = pred_x0 + dir_xt

        return x_i


def train(dataset, n_epoch: int = 100, device: str = 'cpu', sampling: str = 'ddpm') -> None:

    print("Sampling:", sampling)

    eps_model = UNetModel(in_channels=1, out_channels=1)

    print("Num params: ", sum(p.numel() for p in eps_model.parameters()))

    model = DiffusionModel(eps_model=eps_model, betas=(1e-4, 0.02), n_T=1000)
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)

    for i in range(n_epoch):
        model.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = model(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"epoch: {i} - loss: {loss_ema:.4e}")
            optim.step()

        model.eval()
        with torch.no_grad():
            if sampling == 'ddpm':
                xh = model.ddpm_sample(16, (1, 28, 28), device)
                result_folder = './experiments/ddpm'
            elif sampling == 'ddim':
                xh = model.ddim_sample(16, (1, 28, 28), device)
                result_folder = './experiments/ddim'
            
            xh = torch.clamp(xh, -0.5, 0.5) + 0.5
            
            grid = make_grid(xh, nrow=4)
            os.makedirs(result_folder, exist_ok=True)
            save_image(grid, f"{result_folder}/sample_{i}.png")

            # save model
            torch.save(model.state_dict(), f"{result_folder}/{sampling}.pth")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))])
    dataset = MNIST('./dataset', train=True, download=True, transform=tf)
    print("Dataset size:", len(dataset))

    train(dataset=dataset, n_epoch=1000, device=device, sampling='ddim')