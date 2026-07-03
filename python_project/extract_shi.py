'''
This file:
- defines extract_shi function 
- creates raw data files with the latent values that can be used in the graph dataset
'''
import os
import torch
from fc_AE import KSparse
import numpy as np
import tensorflow as tf
from states_check import prepare_simple_dataset

def extract_shi(folders, freq, dataset, GAN_dir=r"C:\Users\Maria\Documents\Honours Programme\Networks\GAN", features=False, path_idx=None):
    '''
    Extract sHI values from the trained autoencoder models for a given dataset and frequency.
    inputs:
    - folders: list of folder names within GAN_dir where the models are stored (usually models after bayesian optimization)
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

    VALIDATION_PANEL_MAP = {'123': '109'}
    model_panel = VALIDATION_PANEL_MAP.get(dataset, dataset)

    for folder in folders:
    
        for file in os.listdir(folder):
            if not file.endswith(".h5"):
                continue
        
        
                
            parts = file.split("_")
            seg = parts[3]
            i = seg.find("-")
            if i != -1:
                path = int(seg[4:i])
            else:
                path = int(seg.split(".")[0])

            
            panel = parts[2]
            if panel != model_panel and features:
                continue
            if path_idx is not None and path != path_idx:
                continue
        
            custom_objs = {"KSparse": KSparse}
            model = tf.keras.models.load_model(os.path.join(folder, file), custom_objects=custom_objs, compile=False, safe_mode=False)
            if features: 
                ds = prepare_simple_dataset(path, freq, dataset, include_benchmark=True, features=True)
            else:
                ds = prepare_simple_dataset(path, freq, dataset, include_benchmark=True)

            if features:
                big_latent_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
            else:
                big_latent_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("k_sparse_1").output)

            _, path_latent = model.predict(ds, verbose=0)
            if path_latent.ndim == 3:
                path_latent = path_latent.reshape(-1, path_latent.shape[-1])

            latents_all.append(path_latent)
            path_labels.append(path)
            big_latent_all.append(big_latent_extractor.predict(ds, verbose=0))

    return latents_all, path_labels, big_latent_all


if __name__ == "__main__":

    folders = ["Model_for_every_path"]
    datasets = ['103', '104', '105', '109','123']
   
    freq = 3
    for dataset in datasets: 
        latents_all, path_labels, big_latent_all = extract_shi(folders, freq, dataset)
        print(f"Collected latents for {len(latents_all)} paths.")
        print(f"Shape of latents matrix: {np.array(latents_all).shape}")
        print(f"Shape of the path labels: {np.array(path_labels).shape}")
        print(f"Big latent shape: {np.array(big_latent_all).shape}")

        out_dict ={"shi": np.array(latents_all), "path_labels": np.array(path_labels), "big_latent": np.array(big_latent_all)}
        folder = r"graph_data\raw"
        torch.save(out_dict, os.path.join(folder, f"panel_{dataset}_shi_raw_{freq}.pt"))