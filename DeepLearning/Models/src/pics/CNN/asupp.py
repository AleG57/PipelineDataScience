import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# -----------------------------
# Paramètres
# -----------------------------
stride = 2  # Modifie ici pour voir l'effet du stride
pad = 1     # Padding 0 (ajoute des pixels artificiels autour)
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)  # Sobel vertical

image = np.array([
    [1, 2, 3, 1, 0, 2],
    [0, 1, 2, 3, 1, 0],
    [3, 1, 0, 2, 2, 1],
    [2, 3, 1, 0, 1, 3],
    [1, 0, 2, 1, 3, 2],
    [2, 1, 3, 0, 2, 1],
], dtype=float)

H, W = image.shape
kH, kW = kernel.shape

# -----------------------------
# Padding à 0
# -----------------------------
padded_image = np.pad(image, pad, mode='constant', constant_values=np.nan)
pH, pW = padded_image.shape

out_H = (pH - kH) // stride + 1
out_W = (pW - kW) // stride + 1
output = np.full((out_H, out_W), np.nan)

# -----------------------------
# Figure 3 panneaux
# -----------------------------
fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
ax_img = fig.add_subplot(gs[0])
ax_calc = fig.add_subplot(gs[1])
ax_out = fig.add_subplot(gs[2])

# --- Image ---
ax_img.set_xlim(0, pW)
ax_img.set_ylim(0, pH)
ax_img.invert_yaxis()
ax_img.set_aspect("equal")
ax_img.axis("off")
ax_img.set_title("Image d'entrée avec padding", fontsize=14, pad=20)

# --- Calcul ---
ax_calc.axis("off")

# --- Sortie ---
ax_out.set_xlim(0, out_W)
ax_out.set_ylim(0, out_H)
ax_out.invert_yaxis()
ax_out.set_aspect("equal")
ax_out.axis("off")
ax_out.set_title(f"Feature map (stride={stride})", fontsize=14, pad=20)

# -----------------------------
# Dessin image
# -----------------------------
for y in range(pH):
    for x in range(pW):
        color = "white" if not np.isnan(padded_image[y, x]) else "lightgrey"
        ls = '-' if not np.isnan(padded_image[y, x]) else '--'
        ax_img.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black", linewidth=1.5, linestyle=ls))
        if not np.isnan(padded_image[y, x]):
            ax_img.text(x + 0.5, y + 0.5, f"{int(padded_image[y, x])}", ha="center", va="center", fontsize=13)
        else:
            ax_img.text(x + 0.5, y + 0.5, "0", ha="center", va="center", fontsize=13, color="grey")

# -----------------------------
# Kernel bleu
# -----------------------------
kernel_cells = []
for _ in range(kH * kW):
    r = Rectangle((0, 0), 1, 1, facecolor="royalblue", alpha=0.4)
    ax_img.add_patch(r)
    kernel_cells.append(r)

# -----------------------------
# Grille sortie
# -----------------------------
out_texts = []
for y in range(out_H):
    for x in range(out_W):
        ax_out.add_patch(Rectangle((x, y), 1, 1, facecolor="white", edgecolor="black", linewidth=1.5))
        t = ax_out.text(x + 0.5, y + 0.5, "·", ha="center", va="center", fontsize=14)
        out_texts.append(t)

# -----------------------------
# Positions (stride)
# -----------------------------
positions = [(y, x) for y in range(0, out_H*stride, stride) for x in range(0, out_W*stride, stride)]

# -----------------------------
# Animation
# -----------------------------
def update(frame):
    y, x = positions[frame]

    # Déplacer le kernel
    idx = 0
    for i in range(kH):
        for j in range(kW):
            kernel_cells[idx].set_xy((x + j, y + i))
            idx += 1

    # Patch courant (NaN → 0)
    patch = padded_image[y:y+kH, x:x+kW].copy()
    patch_calc = np.nan_to_num(patch, nan=0.0)
    value = np.sum(patch_calc * kernel)
    out_y = y // stride
    out_x = x // stride
    output[out_y, out_x] = value

    # Mise à jour sortie
    idx_out = out_y * out_W + out_x
    out_texts[idx_out].set_text(f"{value:.2f}")

    return kernel_cells + out_texts

# -----------------------------
# Création GIF
# -----------------------------
anim = FuncAnimation(fig, update, frames=len(positions), interval=900, repeat=True)
anim.save("conv2d_stride_padding.gif", writer=PillowWriter(fps=1))
plt.close()
