"""
This script is used to find optimal deverb parameters
Knowing the impulse responses of the recording we can compute the reconstruction error and minimize it

Copyright to:
    Katarzyna Augustyn
    Dominika Godzisz
    Bartłomiej Piekarz
"""
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from deverb import get_max_h_matrix, dereverberate
from parameterization import STFT, first_larger_square


MAX_H_TYPES = ["log-log", "log-lin", "log-full", "lin-log", "lin-lin", "lin-full", "const"]
PARAMS = {
    "win_len": 25,
    "win_ovlap": 0.6,
    "blocks": 400,
    "max_h_type": "lin-lin",
    "min_gain_dry": 0,
    "bias": 1,
    "alpha": 0,
    "gamma": 0.3,
}
H_RIRS = ["deverb_test_samples/IMP_short.wav", "deverb_test_samples/IMP_medium.wav", "deverb_test_samples/IMP_long.wav"]
SAMPLES = ["deverb_test_samples/test_short.wav", "deverb_test_samples/test_medium.wav", "deverb_test_samples/test_long.wav"]
plots = True


def read_H_rir(path):
    wave, fs = sf.read(path)
    win_len = int(PARAMS["win_len"] / 1000 * fs)
    win_ovlap = int(PARAMS["win_ovlap"] * win_len)
    window = np.hanning(win_len)

    stft = STFT(wave, window, win_ovlap, first_larger_square(win_len), power=True)
    frames, freqs = stft.shape

    if frames > PARAMS["blocks"]:
        stft = stft[0:PARAMS["blocks"], :]
    elif frames < PARAMS["blocks"]:
        stft = np.vstack((stft, np.zeros((PARAMS["blocks"] - frames, freqs))))

    return stft[:, 0:freqs // 2 + 1]


def plot_max_h_types():
    freqs = 100
    blocks = 50

    fig, axes = plt.subplots(len(MAX_H_TYPES), figsize=(5, 20))
    for i, item in enumerate(MAX_H_TYPES):
        axes[i].pcolormesh(get_max_h_matrix(item, freqs, blocks).T)
        axes[i].set_title(item)
    fig.show()


def plot_errors(x, errors, title, x_label, x_ticks=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    if x_ticks:
        x_t = np.arange(0, errors.size)
        plt.setp(ax, xticks=x_t, xticklabels=x)
        ax.plot(x_t, errors)
    else:
        ax.plot(x, errors)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("error")
    fig.show()


def plot_stfts(org_H, est_H, fs, title):
    hop = PARAMS["win_len"] * (1 - PARAMS["win_ovlap"]) / 1000
    frames, freqs = org_H.shape

    txx = np.linspace(0, hop * frames, frames)
    fxx = np.linspace(0, fs / 2000, freqs)

    Txx, Fxx = np.meshgrid(txx, fxx)

    fig, axes = plt.subplots(2, figsize=(6, 9))
    fig.suptitle(title)
    axes[0].pcolormesh(Txx, Fxx, org_H.T)
    axes[0].set_title(r"original $H_{rir}$")
    axes[0].set_ylabel(r"frequency $[kHz]$")
    axes[1].pcolormesh(Txx, Fxx, est_H.T)
    axes[1].set_title(r"estimated $H_{rir}$")
    axes[1].set_ylabel(r"frequency $[kHz]$")
    axes[1].set_xlabel(r"time $[s]$")
    fig.show()


def compute_error(org_H, est_H):
    org_H_max = np.max(org_H)
    est_H_max = np.max(est_H)
    # dividing by the number of frames for more correct h length error
    return np.sum(np.abs(org_H / org_H_max - est_H / est_H_max)) / org_H.shape[0]


def test_win_len(wave, fs, rir_path):
    print("Testing window lengths")

    def_len = PARAMS["win_len"]

    lens = [5, 10, 15, 20, 25, 30, 40, 50]
    errors = np.zeros(len(lens))

    for i, item in enumerate(lens):
        print("Computing error for window length of {} ms ({}/{})".format(item, i + 1, len(lens)))

        PARAMS["win_len"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for window length of " + str(item) + " ms")

    PARAMS["win_len"] = def_len
    plot_errors(lens, errors, "window length reconstruction errors", r"window length $[ms]$")

    print("")


def test_win_ovlap(wave, fs, rir_path):
    print("Testing window overlaps")

    def_ovlap = PARAMS["win_ovlap"]

    ovlaps = [0.1, 0.25, 0.5, 0.75, 0.9]
    errors = np.zeros(len(ovlaps))

    for i, item in enumerate(ovlaps):
        print("Computing error for window overlap of {} % ({}/{})".format(item, i + 1, len(ovlaps)))

        PARAMS["win_ovlap"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for window overlap of " + str(item) + " %")

    PARAMS["win_ovlap"] = def_ovlap
    plot_errors(ovlaps, errors, "window overlap reconstruction errors", r"window overlap $[\%]$")

    print("")


def test_h_length(wave, fs, rir_path):
    print("Testing response lengths")

    def_blocks = PARAMS["blocks"]

    lens = [100, 200, 400, 800]
    errors = np.zeros(len(lens))

    for i, item in enumerate(lens):
        print("Computing error for h length of {} frames ({}/{})".format(item, i + 1, len(lens)))

        PARAMS["blocks"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for response length of " + str(item) + " frames")

    PARAMS["blocks"] = def_blocks
    plot_errors(lens, errors, "response length reconstruction errors", r"response length")

    print("")


def test_h_type(wave, fs, rir_path):
    print("Testing max h types")

    def_type = PARAMS["max_h_type"]

    errors = np.zeros(len(MAX_H_TYPES))

    for i, item in enumerate(MAX_H_TYPES):
        print("Computing error for h type of {} ({}/{})".format(item, i + 1, len(MAX_H_TYPES)))

        PARAMS["max_h_type"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for max_h type of " + str(item))

    PARAMS["max_h_type"] = def_type
    plot_errors(MAX_H_TYPES, errors, "max h type reconstruction errors", r"max h type", True)

    print("")


def test_min_gain_dry(wave, fs, rir_path):
    print("Testing min dry gains")

    def_gain = PARAMS["min_gain_dry"]

    gains = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
    errors = np.zeros(len(gains))

    for i, item in enumerate(gains):
        print("Computing error for min dry gain of {} ({}/{})".format(item, i + 1, len(gains)))

        PARAMS["min_gain_dry"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for min dry gain of " + str(item))

    PARAMS["min_gain_dry"] = def_gain
    plot_errors(gains, errors, "min dry gain reconstruction errors", r"min dry gain")

    print("")


def test_bias(wave, fs, rir_path):
    print("Testing biases")

    def_bias = PARAMS["bias"]

    bias = [0.5, 0.9, 0.99, 1, 1.01, 1.1, 1.5]
    errors = np.zeros(len(bias))

    for i, item in enumerate(bias):
        print("Computing error for bias of {} ({}/{})".format(item, i + 1, len(bias)))

        PARAMS["bias"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for bias of " + str(item))

    PARAMS["bias"] = def_bias
    plot_errors(bias, errors, "bias reconstruction errors", r"bias")

    print("")


def test_alpha(wave, fs, rir_path):
    print("Testing alphas")

    def_alpha = PARAMS["alpha"]

    alpha = np.linspace(0, 1, 10)
    errors = np.zeros(len(alpha))

    for i, item in enumerate(alpha):
        print("Computing error for alpha of {} ({}/{})".format(item, i + 1, len(alpha)))

        PARAMS["alpha"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for alpha of " + str(item))

    PARAMS["alpha"] = def_alpha
    plot_errors(alpha, errors, "alpha reconstruction errors", r"alpha")

    print("")


def test_gamma(wave, fs, rir_path):
    print("Testing gammas")

    def_gamma = PARAMS["gamma"]

    gamma = np.linspace(0, 1, 10)
    errors = np.zeros(len(gamma))

    for i, item in enumerate(gamma):
        print("Computing error for gamma of {} ({}/{})".format(item, i + 1, len(gamma)))

        PARAMS["gamma"] = item
        org_H_rir = read_H_rir(rir_path)
        est_H_rir, _, _ = dereverberate(wave, fs, PARAMS, False)
        errors[i] = compute_error(org_H_rir, est_H_rir)

        if plots:
            plot_stfts(org_H_rir, est_H_rir, fs, r"$H_{rir}$ for gamma of " + str(item))

    PARAMS["gamma"] = def_gamma
    plot_errors(gamma, errors, "gamma reconstruction errors", r"gamma")

    print("")


if __name__ == "__main__":
    plot_max_h_types()

    sample = 1
    wave, fs = sf.read(SAMPLES[sample])

    # test_win_len(wave, fs, H_RIRS[sample])
    # test_win_ovlap(wave, fs, H_RIRS[sample])
    # test_h_length(wave, fs, H_RIRS[sample])
    # test_h_type(wave, fs, H_RIRS[sample])
    # test_min_gain_dry(wave, fs, H_RIRS[sample])     # irrelevant
    # test_bias(wave, fs, H_RIRS[sample])
    # test_alpha(wave, fs, H_RIRS[sample])
    # test_gamma(wave, fs, H_RIRS[sample])            # irrelevant
