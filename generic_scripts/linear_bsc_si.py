"""Script to calculate the beam stay clear for Sirius storage ring.

BSC was calculated considering linear approximation.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyaccel as pa
from pymodels import si

plt.rcParams.update(
    {
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "axes.grid": True,
        "axes.titlesize": 12,
        "lines.linewidth": 1.8,
    }
)

# creates sirius model
model = si.create_accelerator()
model = pa.lattice.refine_lattice(model)
model.cavity_on = True
model.radiation_on = True
model.vchamber_on = True
famdata = si.get_family_data(model)
tws, *_ = pa.optics.calc_twiss(model)
spos = tws.spos

# calculates linear beam stay clear
bscx, bscy, *_ = pa.optics.acceptances.calc_beam_stay_clear(model, tws)

# creates a .txt file with BSC info
section_name = si.families.get_section_name_mapping(model)
fam_name = []
for elem in model:
    fam_name.append(elem.fam_name)

description = "Horizontal beam stay clear (BSCx) and vertical beam stay clear \
(BSCy)\nfor Sirius storage ring. BSC was calculated considering linear \
approximation.\n"

file_name = "sirius_bsc.txt"

with open(file_name, "w") as fil:
    fil.write(description + "\n")
    fil.write(
        "{:10s} {:15s} {:^11s} {:^11s} {:^11s}\n".format(
            "sec-sub",
            "family name",
            "spos [m]",
            "BSCx [mm]",
            "BSCy [mm]",
        )
    )
    for i, sec in enumerate(section_name):
        fil.write(
            f"{sec:10s} {fam_name[i]:15s} {spos[i]:^11.3f} "
            + f"{bscx[i]*1e3:^11.3f} {bscy[i]*1e3:^11.3f}\n"
        )

# import the data from sirius_bsc.txt to plot BSC figure
data = np.loadtxt(fname="sirius_bsc.txt", dtype="<U21", skiprows=4)

sec = data[:, 0]
fam_name = data[:, 1]
spos = data[:, 2].astype("float")
bscx = data[:, 3].astype("float")
bscy = data[:, 4].astype("float")

# plots the BSC figure for wiki-sirius
symmetry = 10

res = 512 / 370
h = 6
fig, axs = plt.subplots(
    3,
    1,
    figsize=(res * h, h),
    height_ratios=[3.5, 1, 3.5],
    sharex=True,
    gridspec_kw=dict(
        left=0.08, right=0.98, top=0.95, bottom=0.08, hspace=0.02
    ),
)

ax1 = axs[0]
pa.graphics.draw_lattice(model, gca=axs[1], symmetry=symmetry)
ax2 = axs[2]

ax1.plot(spos, bscx, color="blue")
ax1.set_ylabel("Horizontal BSC [mm]")
ax1.set_title(f"Beam Stay Clear - {si.lattice_version}")
ax1.set_ylim(0, None)

ax2.plot(spos, bscy, color="red")
ax2.set_ylabel("Vertical BSC [mm]")
ax2.set_xlabel("s [m]")
ax2.set_ylim(0, None)

ticks = np.linspace(0, 50, 11, dtype="int")
ax1.set_xticks(ticks)
ax2.set_xticks(ticks)

axs[1].set_ylim(-1, 1)
axs[1].axis("off")
axs[0].set_xlim(0, model.length / symmetry)

fig.savefig("si_bsc.svg")

# calculates BSC for Sirius straight sections


def beta_ss(spos, beta0):
    """Calculates beta at the center of a straight section."""
    return beta0 + spos * spos / beta0


def bsc_ss(beta, beta0, bsc0):
    """Calculates BSC at the center of a straight section."""
    return np.sqrt(beta / beta0) * bsc0


symmetry = 20

# center points of the first 3 straight sections (SA, SB, SP)
center_ss = model.length / symmetry * np.arange(3)
dist_center = np.linspace(0, 3, 100)

bscx_ss = []
bscy_ss = []

for pos in center_ss:
    idx = np.searchsorted(spos, pos)
    betax0 = tws.betax[idx]
    betay0 = tws.betay[idx]
    bscx0 = bscx[idx]
    bscy0 = bscy[idx]

    betax = beta_ss(dist_center, betax0)
    betay = beta_ss(dist_center, betay0)

    bscx_ss.append(bsc_ss(betax, betax0, bscx0))
    bscy_ss.append(bsc_ss(betay, betay0, bscy0))

bscx_a = bscx_ss[0]  # BSCx at center of high-beta straight section SA
bscx_b = bscx_ss[1]  # BSCx at center of low-beta straight section SB
bscx_p = bscx_ss[2]  # BSCx at center of low-beta straight section SP

bscy_a = bscy_ss[0]  # BSCy at center of high-beta straight section SA
bscy_b = bscy_ss[1]  # BSCy at center of low-beta straight section SB
bscy_p = bscy_ss[2]  # BSCy at center of low-beta straight section SP

# plots the BSC figure at straight sections for wiki-sirius
res = 512 / 386
h = 6

fig, axs = plt.subplots(
    2,
    2,
    figsize=(res * h, h),
    sharex=True,
    gridspec_kw=dict(
        left=0.12, right=0.96, top=0.95, bottom=0.08, hspace=0.1, wspace=0.14
    )
)

axs = axs.flat

axs[0].plot(dist_center, bscx_a, color="blue")
axs[0].set_ylabel("Horizontal BSC [mm]")
axs[0].set_title("High Beta SS (SA)")

axs[1].plot(dist_center, bscx_b, color="blue")
axs[1].plot(dist_center, bscx_p, color="blue")
axs[1].set_title("Low Beta SS (SB, SP)")

axs[2].plot(dist_center, bscy_a, color="red")
axs[2].set_xlabel("distance from center [m]")
axs[2].set_ylabel("Vertical BSC [mm]")

axs[3].plot(dist_center, bscy_b, color="red")
axs[3].plot(dist_center, bscy_p, color="red")
axs[3].set_xlabel("distance from center [m]")

for ax in axs:
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.set_xlim(dist_center[0], dist_center[-1])
    ticks = ax.get_yticks()
    ax.set_ylim(ticks[0], ticks[-1])

fig.savefig("si_bsc_straight_sec.svg")
