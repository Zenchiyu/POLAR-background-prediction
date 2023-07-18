import numpy as np


def compute_freqs(size, timestep=2):
    # d=2 seconds by default
    freqs_tmp = np.fft.fftfreq(size, d=timestep)
    freqs = freqs_tmp[:size//2+1]
    return freqs

def get_freqs_hour(freq):
    return 3600*freq

def get_freqs_interval(size,
                       low_n=0,
                       high_n=None,
                       timestep=2):
    # Returns from the low_n-th freq. up to high_n-th freq included.
    freqs = compute_freqs(size, timestep=timestep)
    if high_n is None:
        return freqs[low_n:]
    return freqs[low_n:np.minimum(freqs.size, high_n)]


def get_freqs_after_first(size, n=None, timestep=2):
    # if n is not None, we select the n-1 freqs after the first
    return get_freqs_interval(size, low_n=1,
                              high_n=n, timestep=timestep)

def get_freqs(size, n=None, timestep=2):
    # if n is not None, we select the n first freqs
    return get_freqs_interval(size, low_n=0,
                            high_n=n, timestep=timestep)

def f_interval(size,
               low_n=0,
               high_n=None,
               timestep=2):
    return get_freqs_hour(get_freqs_interval(size,
                                             low_n=low_n,
                                             high_n=high_n,
                                             timestep=timestep))

def f_after_first(size, n=None, timestep=2):
    return get_freqs_hour(get_freqs_after_first(size, n,
                                                timestep=timestep))

def f(size, n=None, timestep=2):
    return get_freqs_hour(get_freqs(size, n,
                                    timestep=timestep))