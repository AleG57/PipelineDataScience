import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# -----------------------------
# 1️⃣ Charger Iris
# -----------------------------
iris = load_iris()
X = iris.data

# Standardisation (4 features)
X_scaled = StandardScaler().fit_transform(X)

# PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# 2️⃣ Initialisation k-means
# -----------------------------
k = 3
np.random.seed(42)

indices = np.random.choice(len(X_scaled), k, replace=False)
centroids = X_scaled[indices]

frames = []
os.makedirs("frames", exist_ok=True)

# -----------------------------
# 3️⃣ Scénario d'animation (réduit)
# -----------------------------
animation_steps = [
    {"action": "init",   "repeat": 6, "label": "Initialisation des centroïdes"},
    {"action": "assign", "repeat": 6, "label": "Assignation des points"},
    {"action": "update", "repeat": 6, "label": "Mise à jour des centroïdes"},
    {"action": "assign", "repeat": 6, "label": "Nouvelle assignation"},
    {"action": "update", "repeat": 6, "label": "Nouvelle mise à jour"},
    {"action": "assign", "repeat": 6, "label": "Convergence"},
]

total_steps = sum(step["repeat"] for step in animation_steps)
current_step = 1

labels = np.zeros(len(X_scaled), dtype=int)

for step in animation_steps:

    action = step["action"]
    repeat = step["repeat"]
    label_text = step["label"]

    if action in ["init", "assign"]:
        distances = np.linalg.norm(
            X_scaled[:, None] - centroids[None, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

    if action == "update":
        centroids = np.array([
            X_scaled[labels == i].mean(axis=0)
            for i in range(k)
        ])

    centroids_pca = pca.transform(centroids)

    for _ in range(repeat):

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            hue=labels,
            palette="Set1",
            s=70,
            alpha=0.85,
            legend=False
        )

        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c="black",
            s=260,
            marker="X"
        )

        plt.title(
            f"K-means sur Iris\nÉtape {current_step} / {total_steps}",
            fontsize=14
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.axis("equal")

        fname = f"frames/frame_{current_step:03d}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

        frames.append(fname)
        current_step += 1


with imageio.get_writer(
    "kmeans_iris.gif",
    mode="I",
    duration=0.1,  # lent mais pas long
    loop=0
) as writer:
    for frame in frames:
        writer.append_data(imageio.imread(frame))
