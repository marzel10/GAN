import os

import torch
from states_check import prepare_simple_dataset
from fc_AE import KSparse
import numpy as np
import tensorflow as tf

def extract_shi(folders, freq, dataset, GAN_dir=r"C:\Users\Maria\Documents\Honours Programme\Networks\GAN"):
    
    folders = [os.path.join(GAN_dir, f) for f in folders]
    latents_all = []
    path_labels = []
    big_latent_all = []

    for folder in folders:
    
        for file in os.listdir(folder):
            if not file.endswith(".h5"):
                continue
        
            parts = file.split("_")
            # find first "-" part and take the number after "path"
            seg = parts[3]
            i = seg.find("-")
            path = int(parts[3][4:i])
            custom_objs = {"KSparse": KSparse}
            model = tf.keras.models.load_model(os.path.join(folder, file), custom_objects=custom_objs, compile=False, safe_mode=False)
            ds = prepare_simple_dataset(path, freq, dataset, include_benchmark=True)

            batches = []
            big_latent = []
            big_latent_extractor = tf.keras.Model(inputs=model.input, outputs=model.get_layer("k_sparse_1").output)
                
            for x, y in ds:
                _, latent_batch = model.predict(x, verbose=0)
                # if latent_batch is 3D (batch, time, dim) flatten time into samples:
                if latent_batch.ndim == 3:
                    latent_batch = latent_batch.reshape(-1, latent_batch.shape[-1])
                batches.append(latent_batch)

                latent_batch = big_latent_extractor.predict(x, verbose=0)
                big_latent.append(latent_batch)

            path_latent = np.concatenate(batches, axis=0)   # (n_states, latent_dim)
            latents_all.append(path_latent)
            path_labels.append(path)
            big_latent_all.append(np.concatenate(big_latent, axis=0))

    return latents_all, path_labels, big_latent_all


if __name__ == "__main__":

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