"""Script to calculate the beam stay clear for Sirius storage ring.

BSC was calculated considering linear approximation.
"""

import matplotlib.pyplot as plt
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
