"""
This script estimates H_rirs of recordings from a specific folder

Copyright to:
    Katarzyna Augustyn
    Dominika Godzisz
    Bartłomiej Piekarz
"""
import pickle
import soundfile as sf
import os
import shutil

from deverb import dereverberate


dir_in = "sploty"
dir_out = "out"
MODE_DRY = 0
MODE_WET = 1
MODE_H = 2
MODE = MODE_WET


def check_out_directory(path):
    try:
        if os.path.isdir(path):
            print("Removing previous files")
            shutil.rmtree(path)
        if not os.path.exists(path):
            print("Creating directory: " + path)
            os.makedirs(path)
        else:
            raise OSError("Unexpected file found at: " + path)
    except OSError as e:
        print("Could not prepare {} directory!".format(path))
        print(e)
        exit(1)


def scan_directory(path):
    paths = {}
    gen = os.walk(path)
    root, dirs, _ = next(gen)
    for name in dirs:
        paths[name] = []

    for base, _, files in gen:
        base = os.path.basename(os.path.normpath(base))
        check_out_directory(os.path.join(dir_out, base))
        for name in files:
            paths[base].append(os.path.join(base, name))

    return paths


def load_and_estimate(path):
    path = os.path.join(dir_in, path)
    wave, fs = sf.read(path)
    if MODE == MODE_H:
        H_rir, _, _ = dereverberate(wave, fs, estimate_execution_time=False)
        data = {"fs": fs,
                'frame_count': H_rir.shape[0],
                'freq_count': H_rir.shape[1],
                'h_rir': H_rir}
    elif MODE == MODE_DRY:
        _, dry, _ = dereverberate(wave, fs, estimate_execution_time=False)
        data = {"fs": fs,
                "wave": dry}
    else:
        _, _, reverberant = dereverberate(wave, fs, estimate_execution_time=False)
        data = {"fs": fs,
                "wave": reverberant}
    return data


def save_to_file(path, data):
    if MODE == MODE_H:
        path = os.path.splitext(path)[0] + ".p"
        path = os.path.join(dir_out, path)

        with open(path, "wb") as f:
            pickle.dump(data, f)
    else:
        path = os.path.join(dir_out, path)

        with open(path, "wb") as f:
            sf.write(f, data["wave"], data["fs"])


if __name__ == "__main__":
    file_counter = 0

    print("Preparing output directory")
    check_out_directory(dir_out)

    print("Scanning directory: " + dir_in)
    paths = scan_directory(dir_in)

    for i, directory in enumerate(paths.keys()):
        print("Estimating responses in {} ({}/{})".format(directory, i + 1, len(paths)))

        for j, file in enumerate(paths[directory]):
            print("Processing {} ({}/{})".format(file, j + 1, len(paths[directory])))

            estimated = load_and_estimate(file)
            save_to_file(file, estimated)
            file_counter += 1

    print("Processed {} files".format(file_counter))
