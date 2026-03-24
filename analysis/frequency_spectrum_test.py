import numpy as np
import matplotlib.pyplot as plt


def frequency_spectrum(traj, dt):
    """
    Compute and plot frequency spectrum of trajectory.

    Parameters
    ----------
    traj : ndarray
        shape (timesteps, 2)
        [:,0] = q
        [:,1] = p
    dt : float
        timestep used during simulation
    """

    q = traj[:, 0]
    n = len(q)

    # FFT
    fft_vals = np.fft.fft(q)

    # frequencies
    freqs = np.fft.fftfreq(n, dt)

    # keep only positive frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = np.abs(fft_vals[mask])

    # plot
    plt.figure()
    plt.plot(freqs, spectrum)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Frequency Spectrum of q(t)")
    plt.show()


if __name__ == "__main__":

    # load generated trajectory
    traj = np.load("generated/generated_trajectory.npy")

    print("Trajectory shape:", traj.shape)

    # timestep used in simulation
    DT = 0.05

    frequency_spectrum(traj, DT)