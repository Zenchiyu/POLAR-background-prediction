import numpy as np
import matplotlib.pyplot as plt
from src.frequencies import f, f_after_first, f_interval

def plot_quantity_wrt_freqs(ax,
                            quantity,
                            quantity_name,
                            size,
                            low_n=0,
                            high_n=None,
                            timestep=2):
    # size is the size of the original signal in time domain
    params = {"low_n": low_n,
              "high_n": high_n,
              "timestep": timestep}
    ax.plot(f_interval(size, **params), quantity[low_n:high_n])
    ax.set_xlabel(r"Frequency [$hour^{-1}$]")
    ax.set_ylabel(f"{quantity_name}")
    if (low_n == 0):
        if (high_n is None):
            ax.set_title(f"{quantity_name} for all freqs.")
        else:
            ax.set_title(f"{quantity_name} for {high_n} first freqs.")

    elif (low_n ==1):
        if (high_n is None):
            ax.set_title(f"{quantity_name} for all freqs. except first one.")
        else:
            ax.set_title(f"{quantity_name} for {high_n-1} "+\
                         "first freqs. after first one.")
    else:
        ax.set_title(f"{quantity_name} for freq {low_n} "+\
                     f"up to freq {high_n}")
    


def plot_quantity_w_first_freq(quantity, quantity_name,
                                size, n=20000):
    # n is the number of frequencies to show
    # size is the size of the original signal in time domain
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_quantity_wrt_freqs(axs[0],
                            quantity,
                            quantity_name,
                            size,
                            low_n=0, high_n=None)

    plot_quantity_wrt_freqs(axs[1],
                            quantity,
                            quantity_name,
                            size,
                            low_n=0, high_n=n)
    plt.tight_layout()
    plt.show()

def plot_quantity_wo_first_freq(quantity, quantity_name,
                                 size, n=20000):
    # n-1 is the number of frequencies after first freq. to show
    # size is the size of the original signal in time domain
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_quantity_wrt_freqs(axs2[0],
                            quantity,
                            quantity_name,
                            size,
                            low_n=1, high_n=None)
    
    plot_quantity_wrt_freqs(axs2[1],
                            quantity,
                            quantity_name,
                            size,
                            low_n=1, high_n=n)
    plt.tight_layout()
    plt.show()


def plot_magnitude_w_first_freq(mag, size, n=20000):
    # size is the size of the original signal in time domain
    plot_quantity_w_first_freq(mag, "Magnitude", size, n=n)

def plot_magnitude_wo_first_freq(mag, size, n=20000):
    # size is the size of the original signal in time domain
    plot_quantity_wo_first_freq(mag, "Magnitude", size, n=n)

def plot_phase_w_first_freq(phase, size, n=20000):
    # size is the size of the original signal in time domain
    plot_quantity_w_first_freq(phase, "Phase", size, n=n)

def plot_phase_wo_first_freq(phase, size, n=20000):
    # size is the size of the original signal in time domain
    plot_quantity_wo_first_freq(phase, "Phase", size, n=n)