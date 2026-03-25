import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from model.noise_predictor import NoisePredictor
import config, os
import matplotlib.pyplot as plt

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="default_run")
args = parser.parse_args()

RUN_NAME = args.run_name

RUN_DIR = f"experiments/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs("training_progress", exist_ok=True)
device = torch.device("cpu")


# ****************************
# Loading dataset !
# ****************************

data = np.load(config.DATA_PATH)
data = torch.tensor(data, dtype=torch.float32)

dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# ****************************
# Set Diffusion parameters !
# ****************************

T = 1000

config_dict = {
    "epochs" : config.EPOCHS,
    "lr" : 5e-4,
    "lambda_energy" : config.LAMBDA_ENERGY,
    "T" : T,
    "batch_size" : 64
}

with open(f"{RUN_DIR}/config.json","w") as f:
    json.dump(config_dict, f, indent=4)

beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


# ****************************
# Model Calling Model !
# ****************************

model = NoisePredictor().to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

loss_fn = nn.MSELoss()

epochs = config.EPOCHS
eps = config.EPS

# Energy loss weight
lambda_energy = config.LAMBDA_ENERGY
lambda_dyn = config.LAMBDA_DYNAMICS
lambda_diff = config.LAMBDA_DIFFUSION

# ****************************
# Training loop !
# ****************************

for epoch in range(epochs):

    total_loss = 0
    total_diffusion_loss = 0
    total_energy_loss = 0
    total_dyn_loss = 0

    for batch in loader:

        x0 = batch[0].to(device)

        batch_size = x0.shape[0]

        t = torch.randint(0, T, (batch_size,), device=device)

        noise = torch.randn_like(x0)

        a_bar = alpha_bar[t].view(batch_size,1,1)

        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1-a_bar) * noise

        noise_pred = model(xt, t)

        # ============================
        # Diffusion loss
        # ============================

        diffusion_loss = loss_fn(noise_pred, noise)


        # ============================
        # Energy loss
        # ============================

        q = xt[:,:,0]
        p = xt[:,:,1]

        energy = 0.5 * (q**2 + p**2)
        energy_diff = energy[:,1:] - energy[:,:-1]
        energy_loss = torch.mean(energy_diff**2)

        # ============================
        # Dynamics loss
        # ============================

        dt = config.DT

        dq_dt = (q[:,2:] - q[:,:-2]) / (2*dt)
        dp_dt = (p[:,2:] - p[:,:-2]) / (2*dt)

        p_mid = p[:,1:-1]
        q_mid = q[:,1:-1]

        dyn_loss = torch.mean(torch.square(dq_dt - p_mid) + torch.square(dp_dt + q_mid))
        dyn_loss = dyn_loss / (torch.mean(p_mid**2 + q_mid**2) + eps)
        # ============================
        # Total loss
        # ============================

        loss = lambda_diff * diffusion_loss + lambda_energy * energy_loss + lambda_dyn * dyn_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        total_diffusion_loss += diffusion_loss.item()
        total_energy_loss += energy_loss.item()
        total_dyn_loss += dyn_loss.item()


    print(
        f"Epoch {epoch+1} | "
        f"Total Loss: {total_loss/len(loader):.6f} | "
        f"Diffusion Loss: {total_diffusion_loss/len(loader):.6f} | "
        f"Energy Loss: {total_energy_loss/len(loader):.6f}"
        f"Dynamics: {total_dyn_loss/len(loader):.6f}"

    )

    with open(f"{RUN_DIR}/log.txt","a") as f:
        f.write(
            f"Epoch {epoch+1} | "
            f"Total: {total_loss/len(loader):.6f} | "
            f"Diff: {total_diffusion_loss/len(loader):.6f} | "
            f"Energy: {total_energy_loss/len(loader):.6f} | "
            f"Dynamics: {total_dyn_loss/len(loader):.6f}"
        )


# torch.save(model.state_dict(), "model.pth")
torch.save(model.state_dict(), f"{RUN_DIR}/model.pth")