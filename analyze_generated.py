import numpy as np
import matplotlib.pyplot as plt
# from utils.physics import energy
import config

import argparse

def energy(traj):
    q = traj[:,0]
    p = traj[:,1]
    return 0.5*(q**2 + p**2)

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="default_run")
args = parser.parse_args()

RUN_NAME = args.run_name
RUN_DIR = f"experiments/{RUN_NAME}"

print("\n---------------- DATASET INFORMATION ----------------")

# loading dataset
dataset = np.load(config.DATA_PATH)

print("Dataset shape :", dataset.shape)
print("Datatype      :", dataset.dtype)

num_traj, timesteps, dims = dataset.shape

print("Number of trajectories :", num_traj)
print("Timesteps per traj     :", timesteps)
print("State dimension        :", dims)


print("\n---------------- GENERATED TRAJECTORY ANALYSIS ----------------")

traj = np.load(f"{RUN_DIR}/generated_trajectory.npy")

print("Generated trajectory shape :", traj.shape)

H = energy(traj)

print("\nEnergy statistics")
print("----------------------------------------------------------------")
print("Mean energy :", H.mean())
print("Std energy  :", H.std())
print("Min energy  :", H.min())
print("Max energy  :", H.max())


# energy plot
plt.figure()

plt.plot(H)
plt.xlabel("time step")
plt.ylabel("Energy")
plt.title("Energy vs Time (Generated Trajectory)")

plt.tight_layout()
plt.savefig(f"{RUN_DIR}/generated_energy.png", dpi=300)
plt.close()

print("\nEnergy plot saved to plots/generated_energy.png")