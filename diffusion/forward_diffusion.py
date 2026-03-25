import numpy as np
import matplotlib.pyplot as plt
import os
import config

os.makedirs("plots", exist_ok=True)

dataset = np.load(config.DATA_PATH)
trajectory = dataset[0]


T = config.NOISE_STEPS
beta = np.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)


def diffuse(x0, t):

    noise = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * noise

    return xt


steps = [0, 50, 200, 500, 999]

plt.figure(figsize=(10,6))

for i, step in enumerate(steps):

    xt = diffuse(trajectory, step)

    q = xt[:,0]
    p = xt[:,1]

    plt.subplot(2,3,i+1)
    plt.plot(q, p)
    plt.scatter(q, p, s=5)
    plt.title(f"t = {step}")
    plt.xlabel("q")
    plt.ylabel("p")

plt.tight_layout()

plt.savefig("plots/forward_diffusion.png", dpi=300)

plt.close()

print("Forward diffusion plot saved to plots/forward_diffusion.png")