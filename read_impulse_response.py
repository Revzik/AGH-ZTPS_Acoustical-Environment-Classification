import pickle
import soundfile
import numpy as np
import matplotlib.pyplot as plt

from parameterization import STFT


def compute_stft(path, window_length, overlap, nfft, blocks):
    """
    Computes stft of an reference impulse response used to estimate the one from a recording.

    :param path: path to a soundfile (wav)
    :param window_length: length of a window (samples)
    :param overlap: window overlap (samples)
    :param nfft: number of frequency bins
    :param blocks: number of output frequency blocks. If STFT is shorter, then it will get padded with zeros
    :return: stft of an impulse response
    """
    window = np.hanning(window_length)
    wave, fs = soundfile.read(path)

    tmp_stft = STFT(wave, window, overlap, nfft, power=True)

    K = tmp_stft.shape[0]
    stft = np.zeros((blocks, nfft))
    if K >= blocks:
        stft = tmp_stft[0:blocks, :]
    else:
        stft[0:K, :] = tmp_stft

    return stft[:, 0:nfft // 2 + 1]


def save(path, stft, window_length, overlap, nfft, blocks):
    """
    Saves the response into a file

    :param path: path to save the file to
    :param stft: stft of an impulse response wave
    :param window_length: window_length used
    :param overlap: overlap used
    :param nfft: number of frequency bins used
    :param blocks: number of stft blocks
    """
    impulse_dict = {
        "window_length": window_length,
        "overlap": overlap,
        "nfft": nfft,
        "blocks": blocks,
        "stft": stft
    }

    with open(path, "wb") as f:
        pickle.dump(impulse_dict, f)


if __name__ == "__main__":
    _path_in = "real_impulse_responses/IMP_bedroom.wav"
    _window_length = 1024
    _overlap = 768
    _nfft = 1024
    _blocks = 400

    _path_out = "real_impulse_responses/rir_short.p"

    _stft = compute_stft(_path_in, _window_length, _overlap, _nfft, _blocks)
    save(_path_out, _stft, _window_length, _overlap, _nfft, _blocks)

    # fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    # wave, fs = soundfile.read(_path_in)
    # axes[0].plot(wave)
    # axes[1].pcolormesh(_stft.T)
    # axes[1].set_ylim([0, _stft.shape[1] / 2])
    # fig.show()
