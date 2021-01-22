import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("out/bathroom/bathroom_4.p", "rb") as f:
    H_rir = pickle.load(f)

frames, freqs = H_rir.shape
hop = 0.01
f = np.linspace(0, 44100 / 2000, freqs)
t = np.linspace(0, hop * frames, frames)
fxx, txx = np.meshgrid(f, t)

fig, ax = plt.subplots(figsize=(6, 5))
ax.pcolormesh(txx, fxx, H_rir)
ax.set_title(r"estimated $H_{rir}$")
ax.set_xlabel(r"time $[s]$")
ax.set_ylabel(r"frequency $[kHz]$")
fig.show()