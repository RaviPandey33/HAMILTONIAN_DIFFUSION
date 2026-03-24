import numpy as np
import matplotlib.pyplot as plt
import config

"""
Next work : 
Try calling my own method using partitioned RK method here for exact hamiltonian traj's

"""
# from utils.physics import get_hamiltonian_trajectory, energy
def energy(traj):
    q = traj[:,0]
    p = traj[:,1]
    return 0.5 * (q**2 + p**2)

def dynamics(state):
    q, p = state
    dqdt = p
    dpdt = -q
    return np.array([dqdt, dpdt])
def rk4_step(state, dt):
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)

    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def initial_conditions_for_trajectories():

        r = np.random.uniform(0.5, 2.0)
        theta = np.random.uniform(0, 2*np.pi)

        q0 = r * np.cos(theta)
        p0 = r * np.sin(theta)

        return q0, p0

def get_hamiltonian_trajectory(q0, p0, timesteps, dt):

        traj = np.zeros((timesteps, 2))
        state = np.array([q0, p0])

        for t in range(timesteps):
            traj[t] = state
            state = rk4_step(state, dt)

        return traj


def generate_dataset():
    
    data = np.zeros((config.NUMBER_OF_TRAJECTORIES, config.TIMESTEPS, 2))
    print(data.shape)


    for i in range(config.NUMBER_OF_TRAJECTORIES):
        q_0, p_0 = initial_conditions_for_trajectories()
        trajectories = get_hamiltonian_trajectory(q_0,p_0,config.TIMESTEPS,config.DT)
        # change RK4 trajectories to Hamiltonians
        data[i] = trajectories  

    np.save("data/oscillator_dataset.npy", data)
    print(" Trajectory data saved... : ", data.shape)
    return data



if __name__ == "__main__":
    print("Inside the main method")
    dataset = generate_dataset()