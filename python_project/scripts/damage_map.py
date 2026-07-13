'''
This file plot sHI extracted from the AE and plot them on the panel. 
No elipses are implemented yet, only the heatmap of sHI values on the paths.
'''
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import os
from plot_panel import animate_panel_sHI
from extract_shi import extract_shi
from config import ARCHIVE_DIR, DEFAULT_FREQ_INDEX, CMAP_DIVERGING, CMAP_SEQUENTIAL

freq = DEFAULT_FREQ_INDEX
dataset = '104'
GAN_dir = str(ARCHIVE_DIR)
folders = ["Model_for_every_path"]

latents_all, path_labels, _ = extract_shi(folders, freq, dataset, GAN_dir)

print(f"Collected latents for {len(latents_all)} paths.")
print(f"Example latent shape for first path: {latents_all[0].shape}")


n_states = min(p.shape[0] for p in latents_all)
heat = np.stack([p[:n_states].mean(axis=1) for p in latents_all])  # (n_paths, n_states)

anim, _ = animate_panel_sHI(
    panel_number=int(dataset),
    path_indices=path_labels,
    sHI_matrix=heat,
    interval=300,
    cmap=CMAP_DIVERGING,
    save_path=f"panel_{dataset}_sHI_animation.gif",
)
plt.show()

# Plot heatmap of sHI values across paths and states
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heat, aspect="auto", cmap=CMAP_SEQUENTIAL, interpolation="nearest")
ax.set_xlabel("State index")
ax.set_ylabel("Path")
ax.set_yticks(range(len(path_labels)))
ax.set_yticklabels(path_labels)
fig.colorbar(im, ax=ax, label="Latent value")
plt.tight_layout()
plt.show()

