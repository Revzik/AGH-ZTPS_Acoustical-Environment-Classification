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


def dereverberate(wave, fs):
    """
    Estimates the impulse response in a room the recording took place

    :param wave: 1-D ndarray of wave samples
    :param fs: sampling frequency (int or float)
    :returns: 1-D ndarray of the impulse response
              1-D ndarray of the dry signal
              1-D ndarray of the wet signal
    """
    h_rir = np.zeros(wave.shape)
    wave_dry = np.zeros(wave.shape)
    wave_wet = np.zeros(wave.shape)

    return h_rir, wave_dry, wave_wet
