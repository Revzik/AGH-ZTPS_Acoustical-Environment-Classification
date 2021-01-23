import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("out\\bathroom\\bathroom_4.p", "rb") as f:
    data = pickle.load(f)
    H_rir = data["h_rir"]
    fs = data["fs"]
    frame_count = data["frame_count"]
    freq_count = data["freq_count"]

hop = 0.01
f = np.linspace(0, fs / 2000, freq_count)
t = np.linspace(0, hop * frame_count, frame_count)
fxx, txx = np.meshgrid(f, t)

fig, ax = plt.subplots(figsize=(6, 5))
ax.pcolormesh(txx, fxx, H_rir)
ax.set_title(r"estimated $H_{rir}$")
ax.set_xlabel(r"time $[s]$")
ax.set_ylabel(r"frequency $[kHz]$")
fig.show()
