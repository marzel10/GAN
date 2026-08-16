'''
This file:
- defines extract_shi function 
- creates raw data files with the latent values that can be used in the graph dataset
'''
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import torch
import numpy as np
import tensorflow as tf

from CNN_AE import ExpandLastDim, KSparse, SqueezeLastDim
from ae_cross_validation_helper import ClipLayer, _CompatHeNormal
from AE_train import monotonicity_loss
from create_datastores import prepare_simple_dataset
from config import VALIDATION_PANEL_MAP, GRAPH_DATA_DIR, BASE_PANELS

def extract_shi(folders, freq, dataset, GAN_dir=str(_PROJECT_ROOT)):
    '''
    Extract sHI values from the trained autoencoder models for a given dataset and frequency.
    inputs:
    - folders: list of folder names within GAN_dir where the models are stored (models after bayesian optimization)
    - freq: frequency index to extract sHI for (e.g. 3)
    - dataset: dataset/panel number to extract sHI for (e.g. '104')
    - GAN_dir: base directory where the folders are located

    outputs:
    - latents_all: list of numpy arrays, each containing the sHI values for a path (shape: n_states x 1)
    - path_labels: list of path indices corresponding to each entry in latents_all
    - big_latent_all: list of numpy arrays, each containing the big latent values (the one used for reconstruction) for all paths and states 
    '''
    folders = [os.path.join(GAN_dir, f) for f in folders]
    latents_all = []
    path_labels = []
    big_latent_all = []

    model_panel = VALIDATION_PANEL_MAP.get(dataset, dataset)

    for folder in folders:

        # Extract path number from folder name
        last_part = folder.split("_")[-1]
        path_i = last_part.find("path")
        path = int(last_part[path_i + 4:]) if path_i != -1 else None
        print(f"Extracting sHI for path {path}")

        for file in os.listdir(folder):

            # Only load the ensemble model file, skip other files
            if not file.endswith("ensemble_model.keras"):
                continue

            # Load the model with custom objects defined in other files
            CUSTOM_OBJECTS = {
                "KSparse": KSparse,
                "ExpandLastDim": ExpandLastDim,
                "SqueezeLastDim": SqueezeLastDim,
                "ClipLayer": ClipLayer,
                "monotonicity_loss": monotonicity_loss,
                "HeNormal": _CompatHeNormal,
            }
            model = tf.keras.models.load_model(os.path.join(folder, file), custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)

            # Prepare the dataset without division on batches 
            ds = prepare_simple_dataset(path, freq, dataset, include_benchmark=True)
            # Get the sHI (shape: n_states x 1) and latent space (shape: n_states x latent_dim) values from the model
            sHI, _, latent_space = model.predict(ds, verbose=0)
            
            if sHI.ndim == 3:
                sHI = sHI.reshape(-1, sHI.shape[-1])

            latents_all.append(sHI)
            path_labels.append(path)
            big_latent_all.append(latent_space)

    return latents_all, path_labels, big_latent_all

def pre_compute_AE_output():
    for freq in range(0,6):
        folders = [f"Multi_path_BO_fixed_freq{freq}\\Bayesian_CNN_AE_path{i}" for i in range(0, 28)]
        datasets = BASE_PANELS + ["123"]
    
        for dataset in datasets:
            latents_all, path_labels, big_latent_all = extract_shi(folders, freq, dataset)
            print(f"Collected latents for {len(latents_all)} paths.")
            print(f"Shape of latents matrix: {np.array(latents_all).shape}")
            print(f"Shape of the path labels: {np.array(path_labels).shape}")
            print(f"Big latent shape: {np.array(big_latent_all).shape}")

            out_dict ={"shi": np.array(latents_all), "path_labels": np.array(path_labels), "big_latent": np.array(big_latent_all)}
            folder = GRAPH_DATA_DIR / "raw"
            torch.save(out_dict, os.path.join(folder, f"panel_{dataset}_shi_raw_{freq}.pt"))

if __name__ == "__main__":
    pre_compute_AE_output()
