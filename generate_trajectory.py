import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import config

from model.noise_predictor import NoisePredictor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="default_run")
args = parser.parse_args()

RUN_NAME = args.run_name
RUN_DIR = f"experiments/{RUN_NAME}"

os.makedirs("plots", exist_ok=True)
os.makedirs("generated", exist_ok=True)


device = torch.device("cpu")

T = config.NOISE_STEPS
trajectory_length = 130 # need to cross check this once again before final submission


beta = torch.linspace(1e-4, 0.02, T).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


model = NoisePredictor().to(device)
model.load_state_dict(torch.load(f"{RUN_DIR}/model.pth", map_location=device))
model.eval()


x = torch.randn(1, trajectory_length, 2).to(device)


for t in reversed(range(T)):

    t_tensor = torch.tensor([t], dtype=torch.long).to(device)

    with torch.no_grad():
        noise_pred = model(x, t_tensor)

    a = alpha[t]
    a_bar = alpha_bar[t]
    b = beta[t]

    x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - a_bar)) * noise_pred)

    if t > 0:
        z = torch.randn_like(x)
        x = x + torch.sqrt(b) * z


traj = x.squeeze().cpu().numpy()

np.save(f"{RUN_DIR}/generated_trajectory.npy", traj)


plt.figure(figsize=(6,6))
plt.plot(traj[:,0], traj[:,1])
plt.scatter(traj[:,0], traj[:,1], s=5)

plt.xlabel("q")
plt.ylabel("p")
plt.title("Generated trajectory")
plt.axis("equal")

plt.tight_layout()
plt.savefig(f"{RUN_DIR}/generated_trajectory.png", dpi=300)
plt.show()
plt.close()
print("Generated trajectory saved...")