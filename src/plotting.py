n = 20000  # number of frequencies to show

import numpy as np
import matplotlib.pyplot as plt
from src.frequencies import f, f_after_first


def plot_quantity_w_first_freq(quantity, quantity_name,
                                size, n=20000):
    # n is the number of frequencies to show
    # size is the size of the original signal in time domain
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axs[0].plot(f(size), quantity[:])
    axs[0].set_xlabel(r"Frequency [$hour^{-1}$]")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_title(f"{quantity_name} for all freqs.")

    axs[1].plot(f(size,n=n), quantity[:n])
    axs[1].set_xlabel(r"Frequency [$hour^{-1}$]")
    axs[1].set_title(f"{quantity_name} for {n} first freqs.")
    plt.tight_layout()
    plt.show()

def plot_quantity_wo_first_freq(quantity, quantity_name,
                                 size, n=20000):
    # n-1 is the number of frequencies after first freq. to show
    # size is the size of the original signal in time domain
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axs2[0].plot(f_after_first(size), quantity[1:])
    axs2[0].set_xlabel(r"Frequency [$hour^{-1}$]")
    axs2[0].set_ylabel("Magnitude")
    axs2[0].set_title(f"{quantity_name} for all freqs. except first one")

    axs2[1].plot(f_after_first(size, n=n), quantity[1:n])
    axs2[1].set_xlabel(r"Frequency [$hour^{-1}$]")
    axs2[1].set_title(f"{quantity_name} for {n-1} first freqs. after first one")
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