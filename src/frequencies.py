import numpy as np

# freqs_tmp = np.fft.fftfreq(y.size, d=2)
# freqs = freqs_tmp[:y.size//2+1]

def compute_freqs(size):
    freqs_tmp = np.fft.fftfreq(size, d=2)  # 2 seconds
    freqs = freqs_tmp[:size//2+1]
    return freqs

def get_freqs_hour(freq):
    return 3600*freq

def get_freqs_after_first(size, n=None):
    freqs = compute_freqs(size)
    # if n is not None, we select the n-1 freqs after the first
    if n is None:
        return freqs[1:]
    return freqs[1:np.minimum(freqs.size, n)]

def get_freqs(size, n=None):
    freqs = compute_freqs(size)
    # if n is not None, we select the n first freqs
    if n is None:
        return freqs
    return freqs[:np.minimum(freqs.size, n)]

def f_after_first(size, n=None):
    return get_freqs_hour(get_freqs_after_first(size, n))

def f(size, n=None):
    return get_freqs_hour(get_freqs(size, n))