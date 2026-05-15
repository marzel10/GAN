from fc_AE import build_CNN_variable_block, build_deep_fully_connected_network, build_deep_CNN_network
from states import states
from states_check import prepare_datastores
from results_viz import plot_sHI_vs_RUL, PlotContext
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import csv
import json
import io
import pandas as pd

# Monotonicity loss function
def monotonicity_loss(y_true, y_pred):
    # y_pred shape is (batch, 1); enforce monotonicity along batch order
    
    y_flat = tf.reshape(y_pred, [-1])
    #start = y_true[0] * 10_000
    batch_size = tf.shape(y_flat)[0]
    diff = y_flat[1:] - y_flat[:-1]  # consecutive differences, shape (batch-1,)
    diff = diff +tf.ones(batch_size-1, dtype=tf.float32) - tf.random.normal([batch_size-1], mean=0.0, stddev=0.1)  # Shift to ensure positive values for monotonic increase
    diff = tf.pow(diff, 2)  # Square the differences to penalize negative values more heavily
    # Penalize negative steps (violations of monotonic increase)
    #loss = tf.reduce_mean(tf.square(tf.nn.relu(-diff)))
    loss = tf.reduce_mean(diff)
    #loss *= 100000
    #loss += tf.reduce_mean(tf.square(y_flat[0] - start))  # Penalize deviation from start value
    return loss

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
        results_dir=f"model_train_results/{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        seed=None,
    ):
    
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
            "drop_rate": 0.1,
            "k_sparse": 10,
            "input_size": 4000,
            "num_in": 1,
        }
        model = build_deep_fully_connected_network(params)
        model.summary()
    elif net_type == 'CNN_AE':
        params = {
            "drop_rate": 0.3,
            "k_sparse": 10,
            "filter1": 5,
            "filter2": 2,
            "filter3": 32,
            "filter4": 16,
            "kernel_size1": 20,
            "kernel_size2": 10,
            "kernel_size3": 5,
            "kernel_size4": 3,
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

    # Depending on the type of network, there are different names for the final layer
    recon_name = 'final_1' if net_type == 'CNN_AE' else 'fc_output_1'

    # Assign loss functions and weights for the reconstruction and latent space losses
    Vloss_fun = {recon_name: 'mse', 'fc_latent_1': monotonicity_loss}
    if loss_weights is None:
        Vloss_weights = {recon_name: 1.0, 'fc_latent_1': 1.0}
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

    '''
    Information about the model I would like to save:
    - Model architecture summary (number of layers, types of layers, number of parameters)
    - Training configuration (optimizer, learning rate, loss functions, loss weights)
    - batch size used during training
    - training and validation loss values 
    - final loss value at the end of training
    - number of epochs 
    - file name of the saved model (including the final loss in the name for easy reference)
    Save it in a dictionary (in a pickle format) and txt file for easy access 
    '''

    
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
            


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    net_type = 'CNN_AE' # Options: 'fc_AE', 'CNN_AE'
    basic_panels =["109", "105", "104", "103"]
    final_loss_list = []
    rec_train_loss_list = []
    lat_train_loss_list = []
    rec_val_loss_list = []
    lat_val_loss_list = []

    for panel in basic_panels:
        print(f"Validating using panel {panel}...")
        train_ds_names = [p for p in basic_panels if p != panel]
        val_ds_names = [panel]

        model, history, final_loss, model_info, ds_dict, target_dict, RUL_dict, States_dict, train_ds_names, val_ds_names, test_ds_names, results_dir = model_train(net_type, path_i=0,epochs=100, n_blocks=2, train_ds_names=train_ds_names, val_ds_names=val_ds_names, seed=seed, results_dir=f"model_train_results/{pd.Timestamp.now().strftime('%Y-%m-%d_%H')}/{net_type}_panel_{panel}/")
        
        recon_name = 'final_1' if net_type == 'CNN_AE' else 'fc_output_1'

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
        plt.plot(history.history[f'fc_latent_1_loss'], label='Latent Loss')
        plt.plot(history.history[f'val_fc_latent_1_loss'], label='Val Latent Loss')
        plt.title('Latent Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}loss_curves_{net_type}_{final_loss:.4f}.png")

        rec_train_loss_list.append(history.history[f'{recon_name}_loss'])
        lat_train_loss_list.append(history.history[f'fc_latent_1_loss'])
        rec_val_loss_list.append(history.history[f'val_{recon_name}_loss'])
        lat_val_loss_list.append(history.history[f'val_fc_latent_1_loss'])

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
        save = True

        plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="train", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_train')
        plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="validation", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_val')
        plot_sHI_vs_RUL(model, ctx, state_idx=0, dataset_type="test", show=show, save=save, dir=results_dir, plot_name=f'{net_type}_{final_loss:.4f}_sHI_vs_RUL_test')
        final_loss_list.append(final_loss)

    print("Final losses for each panel:", final_loss_list)

    average_final_loss = np.mean(final_loss_list)
    print(f"Average final loss across panels: {average_final_loss:.4f}")

    #Learning progress:
    plt.figure(figsize=(16, 6))

    for i, panel in enumerate(basic_panels):
        plt.subplot(2, 4, i + 1)
        plt.plot(rec_train_loss_list[i], label='Train')
        plt.plot(rec_val_loss_list[i], label='Val')
        plt.title(f'Recon Loss - Panel {panel}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

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

