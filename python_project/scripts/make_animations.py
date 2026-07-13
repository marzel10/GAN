
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from imagining_alghoritm import animate_panel_sidebyside, extract_sHI_after_GAN
from GCN_train import build_and_save_ensemble, EnsembleGCN
from plot_panel import SENSOR_PAIRS
from weight_matrix import failed_sensor_at
from graph_dataset import features_GraphDataset
import torch_geometric


def plot_sHI_for_every_path(model, panel_number, title="sHI vs State for every path", save_path=None, cmap="nipy_spectral", sensor=None):
    """Plot the per-path sHI trajectory across all states, one line per sensor-pair path.

    If `sensor` is given, the figure gets a second subplot showing only the
    paths that include that sensor.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = features_GraphDataset(root='graph_data', panel_number=panel_number, freq=3)
    num_states = len(dataset)
    num_paths = len(SENSOR_PAIRS)

    sHI = np.full((num_paths, num_states), np.nan)
    loader =  torch_geometric.loader.DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _= model(data.x, data.edge_index, data.batch, data.edge_weight)
            sHI[:, data.y.item()] = out.reshape(-1).cpu().numpy()

    colors = plt.cm.get_cmap(cmap, num_paths)

    if sensor is not None:
        fig, (ax_all, ax_sensor) = plt.subplots(1, 2, figsize=(20, 7))
    else:
        fig, ax_all = plt.subplots(figsize=(12, 7))
        ax_sensor = None

    for path, (a, b) in enumerate(SENSOR_PAIRS):
        ax_all.plot(range(num_states), sHI[path], color=colors(path),
                    marker='o', markersize=2, linewidth=1, label=f"{a}-{b}")

    fail_states = [s for s in range(num_states) if failed_sensor_at(dataset.panel_number, s) is not None]
    if fail_states:
        failed_sensor = failed_sensor_at(dataset.panel_number, fail_states[0])
        ax_all.axvline(fail_states[0], color='black', linestyle='--', alpha=0.7,
                        label=f"sensor {failed_sensor} fails")

    ax_all.set_xlabel("State index")
    ax_all.set_ylabel("sHI")
    ax_all.set_title(title)
    ax_all.legend(ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax_all.grid(True)

    if sensor is not None:
        sensor_paths = [(path, (a, b)) for path, (a, b) in enumerate(SENSOR_PAIRS) if sensor in (a, b)]
        for path, (a, b) in sensor_paths:
            ax_sensor.plot(range(num_states), sHI[path], color=colors(path),
                           marker='o', markersize=2, linewidth=1, label=f"{a}-{b}")

        if fail_states:
            ax_sensor.axvline(fail_states[0], color='black', linestyle='--', alpha=0.7,
                              label=f"sensor {failed_sensor} fails")

        ax_sensor.set_xlabel("State index")
        ax_sensor.set_ylabel("sHI")
        ax_sensor.set_title(f"{title} (paths through sensor {sensor})")
        ax_sensor.legend(ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15))
        ax_sensor.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()


features = True  
#output_dir = r"C:\Users\Maria\Documents\Honours Programme\Networks\GAN\Cross_Validation_Results\2026-07-04-19-40-54"
output_dir = r"C:\Users\Maria\Documents\Honours Programme\GAN\Cross_Validation_Results\2026-07-02-22-41-29"
ensemble_model_path = os.path.join(output_dir, "ensemble_model.pt")
#build_and_save_ensemble(output_dir, ensemble_model_path)

ensemble_model = torch.load(ensemble_model_path, weights_only=False)

plot_sHI_for_every_path(ensemble_model, 103)
plot_sHI_for_every_path(ensemble_model, 104, sensor=8)
plot_sHI_for_every_path(ensemble_model, 105, sensor=8)
plot_sHI_for_every_path(ensemble_model, 109)
plot_sHI_for_every_path(ensemble_model, 123, sensor=3)
animate_panel_sidebyside(panel_number=103, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_103_dp")
animate_panel_sidebyside(panel_number=104, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_104_dp")
animate_panel_sidebyside(panel_number=105, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_105_dp")
animate_panel_sidebyside(panel_number=109, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_109_dp")
animate_panel_sidebyside(panel_number=123, model=ensemble_model, n_pixels=10000, c=0.9, beta=0.5, features=features, transform=None, output_dir=output_dir, file_name="WCPDI_panel_123_dp")
