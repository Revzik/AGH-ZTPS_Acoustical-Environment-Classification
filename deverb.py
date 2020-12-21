"""
This script is used to estimate impulse response in a room the recording was taking place in.

The function responsible for that is  dereverberate
It takes two arguments:
    wave - 1-D ndarray of wave samples
    fs - sampling frequency (int or float)

Copyright to:
    Katarzyna Augustyn
    Dominika Godzisz
    Bartłomiej Piekarz
"""
import numpy as np
import scipy as sp
import pickle

from read_impulse_response import compute_stft
from parameterization import STFT


def dereverberate(wave, fs, expected_response_path="real_impulse_responses/rir_medium.p", preloaded=True):
    """
    Estimates the impulse response in a room the recording took place

    :param wave: 1-D ndarray of wave samples
    :param fs: sampling frequency (int or float)
    :param expected_response_path: path to a file containing expected impulse response (default: rir_medium.p)
    :param preloaded: is the response preloaded (default: True)
    :returns: (h_rir) 1-D ndarray of the impulse response,
              (wave_dry) 1-D ndarray of the dry signal,
              (wave_wet) 1-D ndarray of the wet signal
    """
    if preloaded:
        with open(expected_response_path, "rb") as f:
            ref_imp = pickle.load(f)

        win_len = ref_imp["window_length"]
        ovlap = ref_imp["overlap"]
        nfft = ref_imp["nfft"]
        blocks = ref_imp["blocks"]
        imp = ref_imp["stft"]
    else:
        win_len = 1024
        ovlap = 512
        nfft = 1024
        blocks = 400
        imp = compute_stft(expected_response_path, win_len, ovlap, nfft, blocks)

    h_rir = np.zeros(blocks)
    wave_dry = np.zeros(wave.shape)
    wave_wet = np.zeros(wave.shape)

    return h_rir, wave_dry, wave_wet
