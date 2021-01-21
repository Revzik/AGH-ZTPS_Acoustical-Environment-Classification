"""
This script is used to find optimal deverb parameters
Knowing the impulse responses of the recording we can compute the reconstruction error and minimize it

Copyright to:
    Katarzyna Augustyn
    Dominika Godzisz
    Bartłomiej Piekarz
"""
import numpy as np
import matplotlib.pyplot as plt

from deverb import get_max_h_matrix


MAX_H_TYPES = ["log-log", "log-lin", "log-full", "lin-log", "lin-lin", "lin-full", "const"]
PARAMS = {
    "win_len": 25,
    "win_ovlap": 0.75,
    "blocks": 400,
    "max_h_type": "lin-lin",
    "min_gain_dry": 0,
    "bias": 1.01,
    "alpha": 0.2,
    "gamma": 0.3,
}


def plot_max_h_types():
    freqs = 100
    blocks = 50

    fig, axes = plt.subplots(len(MAX_H_TYPES), figsize=(5, 20))
    for i, item in enumerate(MAX_H_TYPES):
        axes[i].pcolormesh(get_max_h_matrix(item, freqs, blocks))
        axes[i].set_title(item)
    fig.show()


def compute_error(org_H, est_H):
    # TODO: appropriate error computing, other tests
    return np.sum(np.abs(org_H - est_H))


def test_windows():
    pass


def test_h_length():
    pass


def test_h_type():
    pass


def test_min_gain_dry():
    pass


def test_bais():
    pass


def test_alpha():
    alphas = np.linspace(0, 1, 10)
    errors = np.zeros(alphas.shape)

    return alphas, errors


def test_gamma():
    gammas = np.linspace(0, 1, 10)
    errors = np.zeros(gammas.shape)

    return gammas, errors


if __name__ == "__main__":
    plot_max_h_types()
