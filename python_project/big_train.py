"""This file defines the process of training of the autoencoders with cross validation (among 103, 104, 105 and 109)
In order to be able to access real network performance"""

import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import pandas as pd

from fc_AE import build_CNN_variable_block, build_deep_fully_connected_network, build_fc_AE_features, build_CNN_AE_features
from states import states
from states_check import prepare_datastores
from results_viz import plot_sHI_vs_RUL, PlotContext


def monotonicity_loss(y_true, y_pred):
    
    vae_seed = 42
    random.seed(vae_seed)
    tf.random.set_seed(vae_seed)
    np.random.seed(vae_seed)

    y_flat = tf.reshape(y_pred, [-1])
    batch_size = tf.shape(y_flat)[0]

    diff = y_flat[1:] - y_flat[:-1]  # consecutive differences, shape (batch-1,)
    diff = diff +tf.ones(batch_size-1, dtype=tf.float32) *10-  0*tf.random.normal([batch_size-1], mean=0.0, stddev=1.0)  # Shift to ensure positive values for monotonic increase
    diff = tf.pow(diff, 2)  # Square the differences to penalize negative values more heavily
   
    length = tf.cast(tf.shape(diff)[0], tf.float32) 
    print(length)
    baseline = 10**2 * length
    loss = tf.reduce_sum(diff) - baseline  # Subtract baseline to allow for some variability without penalty
    return loss

def mse_excl_benchmark(y_true, y_pred):
    signal_true = y_true[:,:,0] # Assuming that the shape is batch_size x n_features x n_channels and that the benchmark features are in the second channel
    signal_pred = y_pred[:,:,0]
    mse = tf.reduce_mean(tf.square(signal_true - signal_pred))
    return mse

def bench_diff_mse(y_true, y_pred):
    bench_true = y_true[:,:,1] # Assuming that the shape is batch_size x n_features x n_channels and that the benchmark features are in the second channel
    signal_true = y_true[:,:,0]
    diff_true = signal_true - bench_true

    return tf.reduce_mean(tf.square(diff_true - (y_pred[:,:,0] - y_pred[:,:,1])))
def model_train(
        net_type, 
        params=None, 
        learning_rate=0.001, 
        loss_weights=None, 
        epochs=50, 
        base_batch_size=30, 
        test_batch_size=1, 
        path_i=0, 
        frequency_i=3, 
        n_blocks=4,
        train_ds_names=["109", "105", "104"], 
        val_ds_names=["103"], 
        test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"], 
        results_dir=f"model_train_results/multipath",
        seed=None,
    ):
    '''
    This function trains a specified type of autoencoder (fully connected or CNN) on the provided datasets,
    with options for hyperparameters, loss functions, and training configurations. 
    It saves the trained model and its training history, along with detailed information
    about the model and training process, in both an Excel database and a text file for easy reference.
    
    Parameters:
    - net_type: Type of autoencoder to train ('fc_AE' or 'CNN_AE').
    - params: Dictionary of hyperparameters for the model architecture. If None, default parameters will be used.
    - loss_weights: Dictionary specifying the weights for the reconstruction and latent losses. If None, equal weights will be used.
    - n_blocks: Number of blocks for the CNN architecture (only relevant if net_type is 'CNN_AE').

    Outputs:
    - model: The trained Keras model.
    - history: The training history object containing loss values and metrics over epochs.
    - final_loss: The final loss value at the end of training.
    - model_info: A dictionary containing detailed information about the model architecture, training configuration, and results.
    - ds_dict, target_dict, RUL_dict, States_dict: Dictionaries containing metadata about the datasets, targets, RUL values, and states.
    - train_ds_names, val_ds_names, test_ds_names: Lists of dataset names used for training, validation, and testing.
    - results_dir: The directory where the trained model and training history are saved.

    '''
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # establish hyperparameters and build model
    if net_type == 'fc_AE' and params is not None: # If params are provided, use them to build the model; otherwise, use default params
        model = build_deep_fully_connected_network(params)
    elif net_type == 'CNN_AE' and params is not None:
        model = build_CNN_variable_block(params)
    elif net_type == 'fc_AE':
        params = {
            "hidden_layer_size1": 1024,
            "hidden_layer_size2": 512,
            "hidden_layer_size3": 256,
            "hidden_layer_size4": 128,
            "desired_latent_size": 15,
            "drop_rate": 0.0,
            "k_sparse": 10,
            "input_size": 4000,
            "num_in": 1,
        }
        model = build_deep_fully_connected_network(params)
        model.summary()
    elif net_type == 'CNN_AE':
       
        params = {
            "drop_rate": 0.0,
            "k_sparse": 10,
            "filter1": 5,
            "filter2": 2,
            "filter3": 32,
            "filter4": 16,
            "kernel_size1": 100,
            "kernel_size2": 50,
            "kernel_size3": 20,
            "kernel_size4": 10,
            "pool_size1": 10,
            "pool_size2": 5,
            "pool_size3": 5,
            "pool_size4": 3,
            "n_blocks": n_blocks,
            'l2_reg': 1e-4,
        }

        model = build_CNN_variable_block(params)
        model.summary()
        
    Vlearning_rate = learning_rate
    Voptimizer = tf.keras.optimizers.Adam(learning_rate=Vlearning_rate)
    Vepochs = epochs

    # Depending on the type of network(CNN on FC), there are different names for the final layer
    recon_name = 'final_1' if net_type == 'CNN_AE' else 'fc_output_1'

    # Assign loss functions and weights for the reconstruction and latent space losses
    Vloss_fun = {recon_name: 'mse', 'fc_latent_1': monotonicity_loss}
    if loss_weights is None:
        Vloss_weights = {recon_name: 1.0, 'fc_latent_1': 2.0}
    else:
        Vloss_weights = loss_weights

    # Prepare the datasets for training, validation, and testing. The function will return the datasets along with dictionaries containing metadata about the datasets, targets, RUL values, and states.
    if net_type == 'fc_AE':
        train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(path_i, frequency_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names) 
    else:
        train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(path_i, frequency_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names, include_benchmark=True) 

            
    model.compile(
        optimizer=Voptimizer,
        loss=Vloss_fun,
        run_eagerly=False,
        loss_weights=Vloss_weights,
    )

    print("Starting training...")

    # Define callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5,
        ),
    ]

    # Train the model and capture the training history
    history = model.fit(train_dataset, epochs=Vepochs, verbose=1, validation_data=val_dataset, callbacks=callbacks)
    final_loss = history.history['loss'][-1]

    # Save the trained model
    data_time = pd.Timestamp.now().strftime("%d-%m-%H-%M")
    if net_type == 'fc_AE':
        path = f"{results_dir}-{data_time}-deep_fully_connected_autoencoder{final_loss:.4f}.h5"
    else:
        path = f"{results_dir}-{data_time}-deep_CNN_autoencoder{final_loss:.4f}.h5"

    model.save(path)
    print(f"Model saved as {path}")
    
    # 1. Capture model summary as a string for the text file
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))

    # 2. Get the number of trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # 3. Create the info dictionary
    model_info = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "net_type": net_type,
        "final_loss": float(f"{final_loss:.6f}"),
        "val_rec_loss": float(f"{history.history[f'val_{recon_name}_loss'][-1]:.6f}"),
        "val_lat_loss": float(f"{history.history['val_fc_latent_1_loss'][-1]:.6f}"),
        "epochs": Vepochs,
        "batch_size": base_batch_size,
        "learning_rate": Vlearning_rate,
        "k_sparse": params.get("k_sparse", "N/A"),
        "trainable_params": trainable_count,
        "hidden_layer_size1": params.get("hidden_layer_size1", "N/A"),
        "hidden_layer_size2": params.get("hidden_layer_size2", "N/A"),
        "hidden_layer_size3": params.get("hidden_layer_size3", "N/A"),
        "hidden_layer_size4": params.get("hidden_layer_size4", "N/A"),
        "desired_latent_size": params.get("desired_latent_size", "N/A"),
        "drop_rate": params.get("drop_rate", "N/A"),
        "filter1": params.get("filter1", "N/A"),
        "filter2": params.get("filter2", "N/A"),
        "filter3": params.get("filter3", "N/A"),
        "filter4": params.get("filter4", "N/A"),
        "kernel_size1": params.get("kernel_size1", "N/A"),
        "kernel_size2": params.get("kernel_size2", "N/A"),
        "kernel_size3": params.get("kernel_size3", "N/A"),
        "kernel_size4": params.get("kernel_size4", "N/A"),
        "pool_size1": params.get("pool_size1", "N/A"),
        "pool_size2": params.get("pool_size2", "N/A"),
        "pool_size3": params.get("pool_size3", "N/A"),
        "pool_size4": params.get("pool_size4", "N/A"),
        "n_blocks": params.get("n_blocks", "N/A"),
        "optimizer": str(Voptimizer.__class__.__name__),
        "loss_weights": str(Vloss_weights),
        "path_index": path_i,
        "frequency_index": frequency_i,
        "train_ds_names": train_ds_names,
        "seed": seed,
        "model_file": path
    }

    # --- SAVE TO EXCEL DATABASE ---
    excel_path = f"model_database.xlsx"
    
    # Create a DataFrame from the single run
    df_new = pd.DataFrame([model_info])

    if not os.path.exists(excel_path):
        # If file doesn't exist, create it
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}")
    else:
        # If exists, append the new row
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Load existing data to find the next empty row
            try:
                existing_df = pd.read_excel(excel_path)
                df_new.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
            except Exception as e:
                # Fallback if file is corrupted/empty
                df_new.to_excel(excel_path, index=False)
        print(f"Updated database at {excel_path}")

    return model, history, final_loss, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir
            
def model_train_features(
        net_type='fc_AE',         # 'fc_AE' or 'CNN_AE'
        include_benchmark=False,
        params=None,
        learning_rate=0.001,
        loss_weights=None,
        epochs=50,
        base_batch_size=30,
        test_batch_size=1,
        path_i=0,
        frequency_i=3,
        train_ds_names=["109", "105", "104"],
        val_ds_names=["103"],
        test_ds_names=["123_1", "123_31", "123_41", "123_43","123_2", "123_32", "123_42", "123_44"],
        results_dir=f"model_train_results/features",
        seed=None,
    ):
    '''
    Trains a features-based autoencoder (FC or CNN) on precomputed time-frequency features.
    net_type: 'fc_AE' uses build_fc_AE_features (flat input),
              'CNN_AE' uses build_CNN_AE_features (2-D feature × channel input).
    include_benchmark: if True, signal+benchmark features are concatenated (doubles n_channels for CNN).
    '''

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    N_FEAT     = 33
    n_channels = 2 if (include_benchmark and net_type == 'CNN_AE') else 1
    flat_size  = N_FEAT * n_channels          # 33 or 66

    # When diff_bench is active (fc_AE + benchmark), the datastore pre-computes
    # signal - benchmark and returns a flat (N_FEAT,) vector, not (N_FEAT, 2).
    enable_diff = (net_type == 'fc_AE' and include_benchmark)
    model_input_size = N_FEAT if enable_diff else flat_size

    # Build model
    if net_type == 'fc_AE':
        default_params = {"k_sparse": 10, "input_size": model_input_size, 'n_features': N_FEAT}
        if params is None:
            params = default_params
        else:
            params["input_size"] = model_input_size   # must match datastore output shape
            params.setdefault("n_features", N_FEAT)
        model = build_fc_AE_features(params)

    elif net_type == 'CNN_AE':
        default_params = {"k_sparse": 10, "n_features": N_FEAT, "n_channels": n_channels,
                          "filters": 16, "latent_dim": 16}
        if params is None:
            params = default_params
        else:
            params.setdefault("n_features", N_FEAT)
            params.setdefault("n_channels", n_channels)
        model = build_CNN_AE_features(params)

    else:
        raise ValueError(f"Unknown net_type '{net_type}'. Choose 'fc_AE' or 'CNN_AE'.")

    Vlearning_rate = learning_rate
    Voptimizer = tf.keras.optimizers.Adam(learning_rate=Vlearning_rate)
    Vepochs = epochs

    Vloss_fun = {'reconstruction': 'mse', 'sHI': monotonicity_loss}
    Vloss_weights = loss_weights if loss_weights is not None else {'reconstruction': 1.0, 'sHI': 2.0}

    train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(
        path_i, frequency_i, base_batch_size, test_batch_size,
        train_ds_names, val_ds_names, test_ds_names,
        include_benchmark=include_benchmark, features=True, diff_bench=enable_diff
    )

    #
            
    model.compile(
        optimizer=Voptimizer,
        loss=Vloss_fun,
        run_eagerly=False,
        loss_weights=Vloss_weights,
    )

    print("Starting training...")

    # Define callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=105, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5,
        ),
    ]

    # Train the model and capture the training history
    history = model.fit(train_dataset, epochs=Vepochs, verbose=1, validation_data=val_dataset, callbacks=callbacks)

    final_loss = np.min(history.history['loss'])  # Use minimum loss achieved during training for naming and logging
    final_loss_idx = np.argmin(history.history['loss'])
    rec_loss_at_final = history.history['reconstruction_loss'][final_loss_idx]
    lat_loss_at_final = history.history['sHI_loss'][final_loss_idx]

    #validation losses at the same epoch as final training loss
    val_final_loss = history.history['val_loss'][final_loss_idx]
    val_rec_loss_at_final = history.history['val_reconstruction_loss'][final_loss_idx]
    val_lat_loss_at_final = history.history['val_sHI_loss'][final_loss_idx]


    # Save the trained model
    data_time = pd.Timestamp.now().strftime("%d-%m-%H-%M")
    bench_tag = "_benchmark" if include_benchmark else ""
    path = f"{results_dir}-{data_time}-{net_type}_features{bench_tag}_{final_loss:.4f}.h5"
   
    model.save(path)
    print(f"Model saved as {path}")
    
    # 1. Capture model summary as a string for the text file
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))

    # 2. Get the number of trainable parameters
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    # 3. Create the info dictionary
    model_info = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "net_type": f"{net_type}_features{'_benchmark' if include_benchmark else ''}",
        "final_loss": float(f"{final_loss:.6f}"),
        "val_rec_loss": float(f"{history.history[f'val_reconstruction_loss'][-1]:.6f}"),
        "val_lat_loss": float(f"{history.history['val_sHI_loss'][-1]:.6f}"),
        "epochs": Vepochs,
        "batch_size": base_batch_size,
        "learning_rate": Vlearning_rate,
        "k_sparse": params.get("k_sparse", "N/A"),
        "trainable_params": trainable_count,
        "hidden_layer_size1": params.get("hidden_layer_size1", "N/A"),
        "hidden_layer_size2": params.get("hidden_layer_size2", "N/A"),
        "hidden_layer_size3": params.get("hidden_layer_size3", "N/A"),
        "hidden_layer_size4": params.get("hidden_layer_size4", "N/A"),
        "desired_latent_size": params.get("desired_latent_size", "N/A"),
        "drop_rate": params.get("drop_rate", "N/A"),
        "filter1": params.get("filter1", "N/A"),
        "filter2": params.get("filter2", "N/A"),
        "filter3": params.get("filter3", "N/A"),
        "filter4": params.get("filter4", "N/A"),
        "kernel_size1": params.get("kernel_size1", "N/A"),
        "kernel_size2": params.get("kernel_size2", "N/A"),
        "kernel_size3": params.get("kernel_size3", "N/A"),
        "kernel_size4": params.get("kernel_size4", "N/A"),
        "pool_size1": params.get("pool_size1", "N/A"),
        "pool_size2": params.get("pool_size2", "N/A"),
        "pool_size3": params.get("pool_size3", "N/A"),
        "pool_size4": params.get("pool_size4", "N/A"),
        "n_blocks": params.get("n_blocks", "N/A"),
        "optimizer": str(Voptimizer.__class__.__name__),
        "loss_weights": str(Vloss_weights),
        "path_index": path_i,
        "frequency_index": frequency_i,
        "train_ds_names": train_ds_names,
        "seed": seed,
        "model_file": path
    }

    # --- SAVE TO EXCEL DATABASE ---
    excel_path = f"model_database.xlsx"
    
    # Create a DataFrame from the single run
    df_new = pd.DataFrame([model_info])

    if not os.path.exists(excel_path):
        # If file doesn't exist, create it
        df_new.to_excel(excel_path, index=False)
        print(f"Created new database: {excel_path}")
    else:
        # If exists, append the new row
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Load existing data to find the next empty row
            try:
                existing_df = pd.read_excel(excel_path)
                df_new.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
            except Exception as e:
                # Fallback if file is corrupted/empty
                df_new.to_excel(excel_path, index=False)
        print(f"Updated database at {excel_path}")

    return model, history, final_loss, rec_loss_at_final, lat_loss_at_final, val_final_loss, val_rec_loss_at_final, val_lat_loss_at_final, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir
            



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    net_type = 'fc_AE' # Options: 'fc_AE', 'CNN_AE'
    basic_panels =["109", "105", "104", "103"] # panel that is used for validation 
    final_loss_list = []
    rec_train_loss_list = []
    lat_train_loss_list = []
    rec_val_loss_list = []
    lat_val_loss_list = []
    features = True
    benchmark = True
    p_idx =0
    path_indexes = np.arange(0, 28)  # Example path indexes to iterate over; adjust as needed
    losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    rec_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    lat_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}

    val_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    val_rec_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}
    val_lat_losses_of_paths = {panel: np.zeros(len(path_indexes)) for panel in basic_panels}

    average_final_loss = np.zeros(len(path_indexes))
    val_average_final_loss = np.zeros(len(path_indexes))

    for p_idx in path_indexes:
        if len(path_indexes) == 1:

            for panel in basic_panels:
                print(f"Validating using panel {panel}...")
                train_ds_names = [p for p in basic_panels if p != panel]
                val_ds_names = [panel]

                if features:
                    params = {
                        "k_sparse": 10,
                        # input_size is overridden by model_train_features to match the datastore shape
                    }
                    model, history, final_loss, rec_loss_at_final, lat_loss_at_final, val_final_loss, val_rec_loss_at_final, val_lat_loss_at_final, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir = model_train_features(
                        net_type='fc_AE',         # switch to 'CNN_AE' to use the CNN variant
                        include_benchmark=benchmark,
                        epochs=200, learning_rate=0.001,
                        params=params, path_i=p_idx, train_ds_names=train_ds_names, val_ds_names=val_ds_names,
                        seed=seed, results_dir=f"model_train_results/features/",
                    )
                    recon_name = 'reconstruction'
                    latent_name = 'sHI'
                else:
                    model, history, final_loss, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir = model_train(net_type, path_i=0, epochs=100, n_blocks=2, train_ds_names=train_ds_names, val_ds_names=val_ds_names, seed=seed, results_dir=f"model_train_results/{pd.Timestamp.now().strftime('%Y-%m-%d_%H')}/{net_type}_panel_{panel}/")
                    
                    recon_name = 'final_1' if net_type == 'CNN_AE' else 'fc_output_1'
                    latent_name = 'fc_latent_1'

                rec_train_loss_list.append(history.history[f'{recon_name}_loss'])
                lat_train_loss_list.append(history.history[f'{latent_name}_loss'])
                rec_val_loss_list.append(history.history[f'val_{recon_name}_loss'])
                lat_val_loss_list.append(history.history[f'val_{latent_name}_loss'])

                
                # print history keys
                print("History keys:", history.history.keys())
                # Plot loss curves
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history.history[f'{recon_name}_loss'], label='Reconstruction Loss')
                plt.plot(history.history[f'val_{recon_name}_loss'], label='Val Reconstruction Loss')
                plt.title('Reconstruction Loss Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(history.history[f'{latent_name}_loss'], label='Latent Loss')
                plt.plot(history.history[f'val_{latent_name}_loss'], label='Val Latent Loss')
                plt.title('Latent Loss Over Epochs')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{results_dir}loss_curves_{net_type}_{final_loss:.4f}.png")

                # Visualize sHI vs RUL for the test dataset
                ctx = PlotContext(
                    train_ds_names=train_ds_names,
                    validation_ds_names=val_ds_names,
                    test_ds_names=test_ds_names,
                    ds_dict=ds_dict,
                    target_dict=target_dict,
                    RUL_dict=RUL_dict,
                    States_dict=States_dict
                )
                
                show = False
                save = False

                plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="train", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_train')
                plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="validation", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_val')
                plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="test", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_test')
                # if not show: 
                #     #clear the plots to save memory
                #     plt.close('all')
                final_loss_list.append(final_loss)

            print("Final losses for each panel:", final_loss_list)

            average_final_loss = np.mean(final_loss_list)
            print(f"Average final loss across panels: {average_final_loss:.4f}")

            #Learning progress:
            plt.figure(figsize=(16, 6))

            # Plot reconstruction loss curves for all panels (on separate subplots)
            for i, panel in enumerate(basic_panels):
                plt.subplot(2, 4, i + 1)
                plt.plot(rec_train_loss_list[i], label='Train')
                plt.plot(rec_val_loss_list[i], label='Val')
                plt.title(f'Recon Loss - Panel {panel}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

            # Plot latent loss curves for all panels (on separate subplots)
            for i, panel in enumerate(basic_panels):
                plt.subplot(2, 4, i + 5)
                plt.plot(lat_train_loss_list[i], label='Train')
                plt.plot(lat_val_loss_list[i], label='Val')
                plt.title(f'Latent Loss - Panel {panel}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

            plt.tight_layout()
            plt.savefig(f"{results_dir}learning_curves_{net_type}_{average_final_loss:.4f}.png")
            plt.show()

            # check dialations 
            # check multiple kernels 
            # number of blocks as a hyperparameter
            # add maxpooling
        else:

            final_loss_list = np.zeros(len(basic_panels))
            val_final_loss_list = np.zeros(len(basic_panels))

            for i, panel in enumerate(basic_panels):
                print(f"Validating using panel {panel}...")
                train_ds_names = [p for p in basic_panels if p != panel]
                val_ds_names = [panel]

                if features:
                    params = {
                        "k_sparse": 10,
                        # input_size is overridden by model_train_features to match the datastore shape
                    }
                    model, history, final_loss, rec_loss_at_final, lat_loss_at_final, val_final_loss, val_rec_loss_at_final, val_lat_loss_at_final, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir = model_train_features(
                        net_type='fc_AE',         # switch to 'CNN_AE' to use the CNN variant
                        include_benchmark=benchmark,
                        epochs=200, learning_rate=0.001,
                        params=params, path_i=p_idx, train_ds_names=train_ds_names, val_ds_names=val_ds_names,
                        seed=seed, results_dir=f"model_train_results/features/",
                    )
                    recon_name = 'reconstruction'
                    latent_name = 'sHI'
                else:
                    model, history, final_loss, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir = model_train(net_type, path_i=0, epochs=100, n_blocks=2, train_ds_names=train_ds_names, val_ds_names=val_ds_names, seed=seed, results_dir=f"model_train_results/{pd.Timestamp.now().strftime('%Y-%m-%d_%H')}/{net_type}_panel_{panel}/")
                    
                    recon_name = 'final_1' if net_type == 'CNN_AE' else 'fc_output_1'
                    latent_name = 'fc_latent_1'
                    
                # training 
                final_loss_list[i] = final_loss
                losses_of_paths[panel][p_idx] = final_loss 
                rec_losses_of_paths[panel][p_idx] = rec_loss_at_final
                lat_losses_of_paths[panel][p_idx] = lat_loss_at_final 

                # validation
                val_final_loss_list[i] = val_final_loss
                val_losses_of_paths[panel][p_idx] = val_final_loss
                val_rec_losses_of_paths[panel][p_idx] = val_rec_loss_at_final
                val_lat_losses_of_paths[panel][p_idx] = val_lat_loss_at_final


            print("Final losses for each panel:", final_loss_list)

            average_final_loss[p_idx] = np.mean(final_loss_list)
            val_average_final_loss[p_idx] = np.mean(val_final_loss_list)
            
            print(f"Average final loss across panels: {average_final_loss[p_idx]:.4f}")
        
    # save the losses of all paths in a csv file for later analysis
    losses_df = pd.DataFrame({
        'path_index': path_indexes,
        'average_final_loss': average_final_loss,
        'val_average_final_loss': val_average_final_loss,
    })
    losses_df.to_csv(f"{results_dir}losses_of_paths.csv", index=False)

    # save losses per panel and path in separate csv files
    for panel in basic_panels:
        panel_df = pd.DataFrame({
            'path_index': path_indexes,
            'training_loss': losses_of_paths[panel],
            'validation_loss': val_losses_of_paths[panel],
            'training_recon_loss': rec_losses_of_paths[panel],
            'validation_recon_loss': val_rec_losses_of_paths[panel],
            'training_latent_loss': lat_losses_of_paths[panel],
            'validation_latent_loss': val_lat_losses_of_paths[panel],
        })
        panel_df.to_csv(f"{results_dir}losses_panel_{panel}.csv", index=False)
    if len(path_indexes) > 1:
        # Plot final losses across panels for each path index
        plt.figure(figsize=(10, 6))
        
        plt.plot(path_indexes, average_final_loss, marker='o', label=f'Path Index {p_idx}')
        plt.plot(path_indexes, val_average_final_loss, marker='s', label=f'Path Index {p_idx} (Validation)')
        plt.title('Final Loss Across Panels for Each Path Index')
        plt.xlabel('Panel Used for Validation')
        plt.ylabel('Final Loss')
        plt.legend()
        plt.show()

        # Plot 4 subplots for every validation panel, on each plot there is training and validation performance for every path 
        # 3 such figures are created: total loss, reconstruction loss and latent loss
        fig_total, axs_total = plt.subplots(2, 2, figsize=(12, 10))
        fig_rec, axs_rec = plt.subplots(2, 2, figsize=(12, 10))
        fig_lat, axs_lat = plt.subplots(2, 2, figsize=(12, 10)) 
        for i, panel in enumerate(basic_panels):
            row, col = divmod(i, 2)
            axs_total[row, col].scatter(path_indexes, losses_of_paths[panel], marker='o', label='Training Loss')
            axs_total[row, col].scatter(path_indexes, val_losses_of_paths[panel], marker='s', label='Validation Loss')
            axs_total[row, col].set_title(f'Panel {panel} - Total Loss')
            axs_total[row, col].set_xlabel('Path Index')
            axs_total[row, col].set_ylabel('Loss')
            axs_total[row, col].legend()

            axs_rec[row, col].scatter(path_indexes, rec_losses_of_paths[panel], marker='o', label='Training Recon Loss')
            axs_rec[row, col].scatter(path_indexes, val_rec_losses_of_paths[panel], marker='s', label='Validation Recon Loss')
            axs_rec[row, col].set_title(f'Panel {panel} - Reconstruction Loss')
            axs_rec[row, col].set_xlabel('Path Index')
            axs_rec[row, col].set_ylabel('Loss')
            axs_rec[row, col].legend()

            axs_lat[row, col].scatter(path_indexes, lat_losses_of_paths[panel], marker='o', label='Training Latent Loss')
            axs_lat[row, col].scatter(path_indexes, val_lat_losses_of_paths[panel], marker='s', label='Validation Latent Loss')
            axs_lat[row, col].set_title(f'Panel {panel} - Latent Loss')
            axs_lat[row, col].set_xlabel('Path Index')
            axs_lat[row, col].set_ylabel('Loss')
            axs_lat[row, col].legend()

        plt.show()

            







