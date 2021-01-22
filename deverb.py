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
import time

from parameterization import STFT, iSTFT, optimal_synth_window, first_larger_square


DEF_PARAMS = {
    "win_len": 25,
    "win_ovlap": 0.75,
    "blocks": 400,
    "min_gain_dry": 0,
    "max_h_type": "lin-lin",
    "bias": 1.01,
    "alpha": 0.2,
    "gamma": 0.3,
}


def get_max_h_matrix(type, freqs, blocks):
    if type == "log-log":
        return np.logspace(np.ones(freqs), np.ones(freqs) * np.finfo(np.float32).eps, blocks) * \
               np.logspace(np.ones(blocks), np.ones(blocks) * np.finfo(np.float32).eps, freqs).T
    elif type == "log-lin":
        return np.logspace(np.ones(freqs), np.ones(freqs) * np.finfo(np.float32).eps, blocks) * \
               np.linspace(np.ones(blocks), np.zeros(blocks), freqs).T
    elif type == "log-full":
        return np.logspace(np.ones(freqs), np.ones(freqs) * np.finfo(np.float32).eps, blocks)
    elif type == "lin-log":
        return np.logspace(np.ones(freqs), np.ones(freqs) * np.finfo(np.float32).eps, blocks) * \
               np.logspace(np.ones(blocks), np.ones(blocks) * np.finfo(np.float32).eps, freqs).T
    elif type == "lin-lin":
        return np.linspace(np.ones(freqs), np.zeros(freqs), blocks) * \
               np.linspace(np.ones(blocks), np.zeros(blocks), freqs).T
    elif type == "lin-full":
        return np.linspace(np.ones(freqs), np.zeros(freqs), blocks)
    else:
        return np.ones((freqs, blocks)).T


def reconstruct(stft, window, overlap):
    frame_count, frequency_count = stft.shape
    sym_stft = np.hstack((stft, np.flipud(np.conj(stft[:, 0:frequency_count - 2]))))
    signal = np.real(iSTFT(sym_stft, window, overlap))
    return signal / np.max(np.abs(signal))


def dereverberate(wave, fs, params=None, estimate_execution_time=True):
    """
    Estimates the impulse response in a room the recording took place

    :param wave: 1-D ndarray of wave samples
    :param fs: int - sampling frequency
    :param params: dict containing the algorithm parameters - keys:
    :param estimate_execution_time: should we print estimated execution time for each next frame
    :returns: (h_stft_pow) 2-D ndarray power STFT of h_rir,
              (wave_dry) 1-D ndarray of the dry signal,
              (wave_wet) 1-D ndarray of the wet signal
    """
    # estimating execution time
    loop_time = 0

    # =================== Parameters ===================
    if params is None:
        params = DEF_PARAMS

    # ==================== Windowing ===================
    win_len_ms = params["win_len"]
    win_ovlap_p = params["win_ovlap"]

    # ================ Times to samples ================
    win_len = int(win_len_ms / 1000 * fs)
    win_ovlap = int(win_len * win_ovlap_p)
    window = np.hanning(win_len)

    # =================== Signal stft ==================
    nfft = first_larger_square(win_len)
    sig_stft = STFT(wave, window, win_ovlap, nfft)
    sig_stft = sig_stft[:, 0:nfft // 2 + 1]
    frame_count, frequency_count = sig_stft.shape

    # ==================== Constants ===================
    # length of the impulse response
    blocks = params["blocks"]

    # minimum gain of dry signal per frequency
    min_gain_dry = params["min_gain_dry"]

    # maximum impulse response estimate
    max_h = get_max_h_matrix(params["max_h_type"], frequency_count, blocks)

    # bias used to keep magnitudes from getting stuck on a wrong minimum
    bias = params["bias"]

    # alpha and gamma - smoothing factors for impulse response magnitude and gain
    alpha = params["alpha"]
    gamma = params["gamma"]

    # ==================== Algorithm ===================
    # dry_stft and wet_stft are the estimated dry and reverberant signals in frequency-time domain
    dry_stft = np.zeros((frame_count, frequency_count), dtype=np.csingle)
    wet_stft = np.zeros((frame_count, frequency_count), dtype=np.csingle)

    # h_stft_pow is the estimated impulse response in frequency-time domain
    h_stft_pow = max_h / 2

    # matrices with the information of currently estimated raw and dry signal (power spectra)
    raw_frames = np.ones((blocks, frequency_count))
    dry_frames = np.zeros((blocks, frequency_count))

    # c is a matrix to keep the raw estimated powers of the impulse response
    c = np.zeros((blocks, frequency_count))

    # gain_dry and gain_wet are the frequency gains of the dry and wet signals
    gain_dry = np.ones(frequency_count)
    gain_wet = np.zeros(frequency_count)

    for i in range(frame_count):
        if estimate_execution_time:
            remaining = loop_time * (frame_count - i)
            print("Processing frame {} of {}, estimated time left: {} ms".format(i + 1, frame_count, remaining))
            loop_time = time.time()

        frame = sig_stft[i, :]
        frame_power = np.power(np.abs(frame), 2)

        # estimate signals based on i-th frame
        for b in range(blocks):

            estimate = frame_power / raw_frames[b, :]
            np.place(estimate, estimate >= h_stft_pow[b, :], h_stft_pow[b, :] * bias + np.finfo(np.float32).eps)
            np.fmin(estimate, max_h[b, :], out=c[b, :])
            h_stft_pow[b, :] = alpha * h_stft_pow[b, :] + (1 - alpha) * c[b, :]

        # calculating gains
        new_gain_dry = 1 - np.sum(dry_frames * h_stft_pow, axis=0) / frame_power
        np.place(new_gain_dry, new_gain_dry < min_gain_dry, min_gain_dry)
        gain_dry = gamma * gain_dry + (1 - gamma) * new_gain_dry

        new_gain_wet = 1 - gain_dry
        gain_wet = gamma * gain_wet + (1 - gamma) * new_gain_wet

        # calculatnig signals
        dry_stft[i, :] = gain_dry * frame
        wet_stft[i, :] = gain_wet * frame

        # shifting previous frames
        dry_frames[1:blocks, :] = dry_frames[0:blocks - 1, :]
        dry_frames[0, :] = np.power(np.abs(dry_stft[i, :]), 2)

        raw_frames[1:blocks, :] = raw_frames[0:blocks - 1, :]
        raw_frames[0, :] = frame_power

        if estimate_execution_time:
            loop_time = round(1000 * (time.time() - loop_time))

    window = optimal_synth_window(window, win_ovlap)

    wave_dry = reconstruct(dry_stft, window, win_ovlap)
    wave_wet = reconstruct(wet_stft, window, win_ovlap)

    return h_stft_pow, wave_dry, wave_wet
