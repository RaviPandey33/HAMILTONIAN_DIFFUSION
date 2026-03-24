import numpy as np
import matplotlib.pyplot as plt

dataset = np.load("data/oscillator_dataset.npy")
traj = dataset[0]

q = traj[:,0]
p = traj[:,1]

energy = 0.5*(q**2 + p**2)
plt.ylim(energy.mean()-1e-7, energy.mean()+1e-7)
plt.plot(energy)
plt.title("Energy vs time (Dataset trajectory)")
plt.show()

print("Energy mean:", energy.mean())
print("Energy std:", energy.std())

print("Energy mean:", energy.mean())
print("Energy std :", energy.std())