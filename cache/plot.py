#!/usr/bin/env python3

import os
import csv

import sys

sys.path.append("..")

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("bmh")

order = [
    "4090",
    "h100",
]


def getOrderNumber(f):
    for o in range(len(order)):
        if f.startswith(order[o]):
            return o
    return len(order) + 1


lineStyle = {"linewidth": 3, "alpha": 0.8, "markersize": 3, "marker": "x"}


fig, ax = plt.subplots(figsize=(8, 4))
fig2, ax2 = plt.subplots(figsize=(8, 4))


for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt"):
        continue

    with open(filename, newline="") as csvfile:
        style = "P-"
        if filename.endswith("f.txt"):
            style = "o--"

        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        bw = []
        L2bw = []
        for row in csvreader:
            if row[0] == "data" or not row[0].isnumeric():
                continue
            sizes.append(float(row[0]))
            bw.append(float(row[4]))
            L2bw.append(float(row[10]))

        # print(sizes)
        # print(bw)
        print(filename, len(sizes), getOrderNumber(filename))
        ax.plot(
            sizes,
            bw,
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle,
        )
        ax2.plot(
            sizes,
            L2bw,
            label=filename[:-4].upper(),
            color="C" + str(getOrderNumber(filename)),
            **lineStyle,
        )

ax.set_xlabel("Data Vol. per SM")
ax.set_ylabel("Bandwidth (GB/s)")
ax.set_xscale("log", base=2)


ax2.set_xlabel("Data Vol. per SM")
ax2.set_ylabel("Bandwidth (GB/s)")
ax2.set_xscale("log", base=2)

formatter = matplotlib.ticker.FuncFormatter(
    lambda x, pos: "{0:g} kB".format(x) if x < 1024 else "{0:g} MB".format(x / 1024)
)
ax.get_xaxis().set_major_formatter(formatter)
# ax.get_yaxis().set_major_formatter(formatter)

ax2.get_xaxis().set_major_formatter(formatter)
# ax2.get_yaxis().set_major_formatter(formatter)

ax.set_xticks([4, 16, 128, 256, 6 * 1024, 20 * 1024, 40 * 1024, 128 * 1024])

ax2.set_xticks([4, 16, 128, 256, 6 * 1024, 20 * 1024, 40 * 1024, 128 * 1024])

ax.set_xlim((1.8, 256 * 1024))
ax2.set_xlim((1.8, 256 * 1024))

fig.autofmt_xdate()
ax.set_ylim([0, ax.get_ylim()[1] * 1.1])

fig2.autofmt_xdate()
ax2.set_ylim([0, ax2.get_ylim()[1] * 1.1])

ax.legend()
fig.tight_layout()

ax2.legend()
fig2.tight_layout()

plt.show()
fig.savefig("cuda-cache.svg")
