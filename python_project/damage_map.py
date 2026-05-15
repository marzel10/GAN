
import numpy as np
import matplotlib.pyplot as plt
import os
from plot_panel import plot_panel_with_paths, animate_panel_sHI
from extract_shi import extract_shi

freq = 3
dataset = '104'
GAN_dir = r"C:\Users\Maria\Documents\Honours Programme\Networks\GAN"
folders = [
    "Multi_path_2026_05_02-22_14_20",
    "Multi_path_2026_05_03-09_15_32",
    "Multi_path_2026_05_03-15_06_17",
    "Multi_path_2026_05_04-00_57_31",
    "Multi_path_2026_05_04-08_54_04",
    "Multi_path_2026_05_05-11_32_40",
    "Multi_path_2026_05_05-15_51_25",
    "Multi_path_2026_05_06-01_03_28",
    "Multi_path_2026_05_06-10_41_54",
    "Multi_path_2026_05_06-21_24_48",
    "Multi_path_2026_05_07-16_28_36",
]



latents_all, path_labels = extract_shi(folders, freq, dataset, GAN_dir)

print(f"Collected latents for {len(latents_all)} paths.")
print(f"Example latent shape for first path: {latents_all[0].shape}")

plot_panel_with_paths(
    panel_number=int(dataset),
    path_indices=path_labels,
    title=f"Panel {dataset}",
)
plt.tight_layout()
plt.show()

# --- Build the heatmap matrix: rows = paths, cols = state idx ---
# Option A: pick a single latent dimension (e.g. dim 0)
# heat = np.stack([p[:, 0] for p in latents_all])

# Option B: reduce across latent dims (mean is usually most interpretable)
# Pad/truncate so all paths have the same number of states
n_states = min(p.shape[0] for p in latents_all)
heat = np.stack([p[:n_states].mean(axis=1) for p in latents_all])  # (n_paths, n_states)

anim, _ = animate_panel_sHI(
    panel_number=int(dataset),
    path_indices=path_labels,
    sHI_matrix=heat,
    interval=300,
    cmap="plasma",
    save_path=f"panel_{dataset}_sHI_animation.gif",
)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heat, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xlabel("State index")
ax.set_ylabel("Path")
ax.set_yticks(range(len(path_labels)))
ax.set_yticklabels(path_labels)
fig.colorbar(im, ax=ax, label="Latent value")
plt.tight_layout()
plt.show()

