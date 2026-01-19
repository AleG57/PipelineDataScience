import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

# ---------------------------
# Paramètres
# ---------------------------
image_path = "input_image.jpg"
block_size = 4
Ks = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 3, 2, 1]
kmeans_iters = 15

resize_input_to = 256
output_scale = 2
frame_duration = 0.1

gif_path = "vector_quantization_voronoi_boundaries.gif"

# ---------------------------
# Chargement de l'image
# ---------------------------
img = Image.open(image_path).convert("RGB")
img = img.resize((resize_input_to, resize_input_to), Image.BICUBIC)
img_np = np.array(img)

H, W, C = img_np.shape
assert H % block_size == 0 and W % block_size == 0

# ---------------------------
# K-means simple
# ---------------------------
def kmeans(X, K, iters=10):
    rng = np.random.default_rng(0)
    centers = X[rng.choice(len(X), K, replace=False)]

    for _ in range(iters):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        for k in range(K):
            if np.any(labels == k):
                centers[k] = X[labels == k].mean(axis=0)

    return centers, labels

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

nby = H // block_size
nbx = W // block_size

frames = []

# ---------------------------
# Image originale agrandie
# ---------------------------
img_orig_big = Image.fromarray(img_np).resize(
    (W * output_scale, H * output_scale),
    Image.NEAREST
)

Wb, Hb = img_orig_big.size
gap = 20

# ---------------------------
# Boucle sur K décroissant
# ---------------------------
for K in Ks:
    print(f"Quantification avec K = {K}")

    centers, labels = kmeans(X, K, iters=kmeans_iters)

    # Quantification
    Xq = centers[labels]

    # Reconstruction image quantifiée
    img_q = np.zeros_like(img_np)
    idx = 0
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            img_q[i:i+block_size, j:j+block_size] = \
                Xq[idx].reshape(block_size, block_size, 3)
            idx += 1

    # ---------------------------
    # Carte des labels par bloc (2D)
    # ---------------------------
    label_map = labels.reshape(nby, nbx)

    # ---------------------------
    # Image Voronoï = frontières seulement
    # ---------------------------
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

    # fond blanc + frontières noires
    img_voronoi = np.dstack([255 - vor]*3)

    # ---------------------------
    # Agrandissement des images
    # ---------------------------
    img_q_big = Image.fromarray(img_q.astype(np.uint8)).resize(
        (W * output_scale, H * output_scale),
        Image.NEAREST
    )

    img_v_big = Image.fromarray(img_voronoi.astype(np.uint8)).resize(
        (W * output_scale, H * output_scale),
        Image.NEAREST
    )

    # ---------------------------
    # Canvas côte à côte (3 images)
    # ---------------------------
    pair = Image.new(
        "RGB",
        (3 * Wb + 2 * gap, Hb),
        (30, 30, 30)
    )

    pair.paste(img_orig_big, (0, 0))
    pair.paste(img_q_big, (Wb + gap, 0))
    pair.paste(img_v_big, (2 * Wb + 2 * gap, 0))

    # ---------------------------
    # Taux de compression
    # ---------------------------
    orig_bits = H * W * 3 * 8
    coded_bits = num_blocks * np.log2(K)
    rate = orig_bits / coded_bits

    # ---------------------------
    # Titre
    # ---------------------------
    title = f"K = {K}   |   Taux de compression = {rate:.1f}:1"

    title_bar_h = 50
    title_bar = Image.new("RGB", (pair.width, title_bar_h), (0, 0, 0))
    out = Image.new("RGB", (pair.width, Hb + title_bar_h))
    out.paste(title_bar, (0, 0))
    out.paste(pair, (0, title_bar_h))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    draw.text((20, 10), title, fill=(255, 255, 255), font=font)

    # ---------------------------
    # Labels au-dessus des images
    # ---------------------------
    label_y = title_bar_h + 10
    draw.text((20, label_y), "Original", fill=(255, 255, 255), font=font)
    draw.text((Wb + gap + 20, label_y), "Quantifiée (VQ)", fill=(255, 255, 255), font=font)
    draw.text((2 * Wb + 2 * gap + 20, label_y),
              "Cellules de Voronoï (frontières)",
              fill=(255, 255, 255), font=font)

    frames.append(np.array(out))

# ---------------------------
# Sauvegarde du GIF
# ---------------------------
imageio.mimsave(
    gif_path,
    frames,
    duration=frame_duration,
    loop=0
)

print(f"GIF sauvegardé : {gif_path}")
