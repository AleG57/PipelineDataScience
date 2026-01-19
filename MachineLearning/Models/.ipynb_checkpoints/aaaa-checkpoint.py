import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ---------------------------
# Paramètres
# ---------------------------
image_path = "input_image.jpg"
block_size = 4
K = 512
kmeans_iters = 15
resize_input_to = 256

# ---------------------------
# Chargement de l'image
# ---------------------------
img = Image.open(image_path).convert("RGB")
img = img.resize((resize_input_to, resize_input_to), Image.BICUBIC)
img_np = np.array(img)

H, W, C = img_np.shape
nby = H // block_size
nbx = W // block_size

# ---------------------------
# Extraction des blocs
# ---------------------------
blocks = []
for i in range(0, H, block_size):
    for j in range(0, W, block_size):
        block = img_np[i:i+block_size, j:j+block_size].reshape(-1)
        blocks.append(block)

X = np.array(blocks)
num_blocks = X.shape[0]

# ---------------------------
# K-means
# ---------------------------
rng = np.random.default_rng(0)
centers = X[rng.choice(len(X), K, replace=False)]
for _ in range(kmeans_iters):
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists, axis=1)
    for k in range(K):
        if np.any(labels == k):
            centers[k] = X[labels == k].mean(axis=0)

# ---------------------------
# Création image Voronoï (frontières)
# ---------------------------
label_map = labels.reshape(nby, nbx)
vor = np.zeros((H, W), dtype=np.uint8)

for by in range(nby):
    for bx in range(nbx):
        k = label_map[by, bx]
        # frontière droite ?
        if bx < nbx - 1 and label_map[by, bx + 1] != k:
            x = (bx + 1) * block_size
            vor[by*block_size:(by+1)*block_size, x-1:x+1] = 255
        # frontière bas ?
        if by < nby - 1 and label_map[by + 1, bx] != k:
            y = (by + 1) * block_size
            vor[y-1:y+1, bx*block_size:(bx+1)*block_size] = 255

# ---------------------------
# Superposition sur l'image quantifiée
# ---------------------------
img_q = np.zeros_like(img_np)
idx = 0
for i in range(0, H, block_size):
    for j in range(0, W, block_size):
        img_q[i:i+block_size, j:j+block_size] = centers[labels[idx]].reshape(block_size, block_size, 3)
        idx += 1

# créer image finale RGB avec contours noirs
vor_rgb = np.dstack([255 - vor]*3)
img_final = np.maximum(img_q, vor_rgb)  # superposition contours

# ---------------------------
# Affichage
# ---------------------------
plt.figure(figsize=(8, 8))
plt.imshow(img_final)
plt.axis('off')
plt.title(f"Quantification vectorielle avec cellules de Voronoï, K={K}")
plt.savefig(f"voronoi_K={K}.png")
plt.show()
