
from states_check import prepare_datastores
from fc_AE import KSparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from dataclasses import dataclass
from typing import Dict, List, Any
import matplotlib.gridspec as gridspec

@dataclass(frozen=True)
class PlotContext:
    ds_dict: Dict[str, Any]
    target_dict: Dict[str, Any]
    RUL_dict: Dict[str, Any]
    train_ds_names: List[str]
    validation_ds_names: List[str]
    test_ds_names: List[str]
    States_dict: Dict[str, Any]

def plot_standard(model, panel_dataset, state_idx, ctx: PlotContext, name: str, ax_rec_spec=None, ax_lat_spec=None):
    States = ctx.States_dict[name]
    sHI = []
    sample_offset = 0
    benchmark = None
    ax_rec = None
    ax_lat = None
    plotted = False
    plots_were_passed = (ax_rec_spec is not None and ax_lat_spec is not None)

    for i, (x, y) in enumerate(panel_dataset):
        reconstruction, latent = model.predict(x)
        x_arr = np.asarray(x)
        reconstruction_arr = np.asarray(reconstruction)
        latent_arr = np.asarray(latent)
        batch_size = int(latent_arr.shape[0])
        batch_states = np.asarray(States[sample_offset:sample_offset + batch_size])
        
        if benchmark is None:
            benchmark = bool(x_arr.ndim == 3 and x_arr.shape[-1] == 2)

        # Initialize axes only once
        if ax_rec is None:
            if not plots_were_passed:
                fig_rec, ax_rec = plt.subplots(1, 2 if benchmark else 1, figsize=(12, 10))
            else:
                fig = ax_rec_spec.get_gridspec().figure
                if benchmark:
                    # Split this specific grid slot into two
                    gs_inner = ax_rec_spec.subgridspec(1, 2)
                    ax_rec = [fig.add_subplot(gs_inner[0]), fig.add_subplot(gs_inner[1])]
                else:
                    ax_rec = fig.add_subplot(ax_rec_spec)
        
        if (not plotted) and np.any(batch_states == state_idx):
            local_idx = int(np.where(batch_states == state_idx)[0][0])
            if benchmark:
                for idx, suffix in enumerate(["Signal", "Benchmark"]):
                    ax_rec[idx].plot(x_arr[local_idx][:, idx], color='blue', alpha=0.6, label='Input')
                    ax_rec[idx].plot(reconstruction_arr[local_idx][:, idx], color='green', alpha=0.8, label='Predicted')
                    ax_rec[idx].set_title(f"{name}: {suffix}")
                    ax_rec[idx].legend()
            else:
                ax_rec.plot(x_arr[local_idx], color='blue', alpha=0.6, label='Input')
                ax_rec.plot(reconstruction_arr[local_idx], color='green', alpha=0.8, label='Predicted')
                ax_rec.set_title(f"{name} state={state_idx}")
                ax_rec.legend()
            plotted = True

        sHI.extend(latent_arr.reshape(batch_size, -1))
        sample_offset += batch_size

    # --- LATENT PLOTTING ---
    if plots_were_passed:
        ax_lat = ax_lat_spec.get_gridspec().figure.add_subplot(ax_lat_spec)
    else:
        fig_lat, ax_lat = plt.subplots(1, 1, figsize=(10, 6))

    sHI_arr = np.asarray(sHI)
    if sHI_arr.size > 0:
        ax_lat.plot(States, sHI_arr[:, 0], 'o', color='purple', markersize=2, label='Test dataset')
        ax_lat.set_title(f'sHI: {name}')
        ax_lat.set_xlabel('State idx')
        ax_lat.legend()
    
    if not plots_were_passed:
        return fig_rec, fig_lat


def plot_123(model, panel123_datasets, state_idx, ctx: PlotContext, names: List[str], ax_rec_spec=None, ax_lat_spec=None):
    sHI = []
    states = []
    benchmark = None
    ax = None      # Reconstruction axes
    ax_lat = None  # Latent axis
    plotted = False
    plots_were_passed = (ax_rec_spec is not None and ax_lat_spec is not None)

    for j, ds in enumerate(panel123_datasets):
        sample_offset = 0
        ds_name = names[j]
        
        for i, (x, y) in enumerate(ds):
            reconstruction, latent = model.predict(x)
            x_arr = np.asarray(x)
            reconstruction_arr = np.asarray(reconstruction)
            latent_arr = np.asarray(latent)
            batch_size = int(latent_arr.shape[0])
            
            if benchmark is None:
                benchmark = bool(x_arr.ndim == 3 and x_arr.shape[-1] == 2)

            # --- INITIALIZE AXES ONCE ---
            if ax is None:
                if not plots_were_passed:
                    fig, ax = plt.subplots(1, 2 if benchmark else 1, figsize=(12, 10))
                    fig_lat, ax_lat = plt.subplots(1, 1, figsize=(10, 6))
                else:
                    # Use the figure from the spec to add subplots
                    fig_rec = ax_rec_spec.get_gridspec().figure
                    if benchmark:
                        gs_inner = ax_rec_spec.subgridspec(1, 2)
                        ax = [fig_rec.add_subplot(gs_inner[0]), fig_rec.add_subplot(gs_inner[1])]
                    else:
                        ax = fig_rec.add_subplot(ax_rec_spec)
                    
                    fig_lat_fig = ax_lat_spec.get_gridspec().figure
                    ax_lat = fig_lat_fig.add_subplot(ax_lat_spec)

            batch_states = np.asarray(ctx.States_dict[ds_name][sample_offset:sample_offset + batch_size])
            
            if (not plotted) and np.any(batch_states == state_idx):
                local_idx = int(np.where(batch_states == state_idx)[0][0])
                if benchmark:
                    for idx, suffix in enumerate(["Signal", "Benchmark"]):
                        ax[idx].plot(x_arr[local_idx][:, idx], color='blue', alpha=0.6, label='Input')
                        ax[idx].plot(reconstruction_arr[local_idx][:, idx], color='green', alpha=0.8, label='Predicted')
                        ax[idx].set_title(f"{ds_name} state={state_idx} - {suffix}")
                        ax[idx].legend()
                else:
                    ax.plot(x_arr[local_idx], color='blue', alpha=0.6, label='Input')
                    ax.plot(reconstruction_arr[local_idx], color='green', alpha=0.8, label='Predicted')
                    ax.set_title(f"State {state_idx} reconstruction")
                    ax.legend()
                plotted = True

            sHI.extend(latent_arr.reshape(batch_size, -1))
            states.extend(batch_states)
            sample_offset += batch_size

    # --- FINAL LATENT PLOT ---
    sHI_arr = np.asarray(sHI)
    if sHI_arr.size > 0:
        ax_lat.plot(states, sHI_arr[:, 0], 'o', color='purple', markersize=2, label='Test dataset')
        ax_lat.set_title(f'sHI Across States: 123')
        ax_lat.set_xlabel('State idx')
    
    if not plots_were_passed:
        return fig, fig_lat

def plot_sHI_vs_RUL(
    model,
    ctx: PlotContext,
    state_idx: int,
    dataset_type: str = "train", # "test" , "validation" or "train"
    save=False,
    dir="results_plot_sHI/",
    plot_name="sHI_vs_RUL",
    show=False,
    plot_together=True
):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if dataset_type == "train":
        print("Plotting sHI vs RUL for TRAIN datasets...")
        names = ctx.train_ds_names

    elif dataset_type == "validation":
        print("Plotting sHI vs RUL for VALIDATION datasets...")
        names = ctx.validation_ds_names

    elif dataset_type == "test":
        print("Plotting sHI vs RUL for TEST datasets...")
        names = ctx.test_ds_names

    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'train', 'validation', or 'test'.")
    
    names = getattr(ctx, f"{dataset_type}_ds_names")
    n_plots = len([n for n in names if not n.startswith("123")]) + (1 if any(n.startswith("123") for n in names) else 0)
    
    cols = (1 if n_plots == 1 else 3)
    rows = int(np.ceil(n_plots / cols))

    # Create the figure containers
    fig_rec_all = plt.figure(figsize=(6*cols, 5*rows))
    gs_rec = gridspec.GridSpec(rows, cols, figure=fig_rec_all)
    
    fig_lat_all, axes_lat = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes_lat = np.atleast_1d(axes_lat).flatten()

    idx = 0
    plot123_done = False

    for name in names:
        row_idx = idx // cols
        col_idx = idx % cols

        if name[:3] == "123" and not plot123_done:
            panel123_datasets = [ctx.ds_dict[ds_name] for ds_name in names if ds_name.startswith("123")]
            names_123 = [ds_name for ds_name in names if ds_name.startswith("123")]
            if plot_together:
                plot_123(model, panel123_datasets, state_idx, ctx, names_123, 
                         ax_rec_spec=gs_rec[row_idx, col_idx], ax_lat_spec=axes_lat[idx])
            else:
                fig_rec, fig_lat = plot_123(model, panel123_datasets, state_idx, ctx, names_123)
                if save:
                    fig_rec.savefig(f"{dir}\\{name}_reconstruction_123.png")
                    fig_lat.savefig(f"{dir}\\{name}_latent_123.png")
            plot123_done = True
            idx += 1
            

        elif name[:3] != "123": # avoid plotting the panel 123 multiple times
            ds = ctx.ds_dict[name]
            if plot_together:
                plot_standard(model, ctx.ds_dict[name], state_idx, ctx, name, 
                              ax_rec_spec=gs_rec[row_idx, col_idx], ax_lat_spec=axes_lat[idx])
            else:
                fig_rec, fig_lat = plot_standard(model, ds, state_idx, ctx, name)

                if save:
                    fig_rec.savefig(f"{dir}\\{name}_reconstruction_{plot_name}.png")
                    fig_lat.savefig(f"{dir}\\{name}_latent_{plot_name}.png")
            idx += 1
    
    fig_rec_all.tight_layout()
   
    if plot_together:
        if save:
            fig_rec_all.savefig(f"{dir}\\all_{plot_name}_reconstruction.png")
            fig_lat_all.savefig(f"{dir}\\all_{plot_name}_latent.png")

        if show:
            plt.show()
    else:

        # Show all figures created above at once.
        if show:
            plt.show(block=True)



if __name__ == "__main__":

    base_batch_size = 30
    test_batch_size = 1
    path_i = 0
    frequency_i = 3
    train_ds_names = ["103", "104", "105"]
    val_ds_names = [ "109"] 
    test_ds_names = ["123_1","123_2","123_31","123_31", "123_32","123_41",  "123_42", "123_43","123_44"]

    train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(path_i, frequency_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names) 
    traun_dataset_bench, val_dataset_bench, test_dataset_bench, ds_dict_bench, target_dict_bench, RUL_dict_bench, States_dict_bench = prepare_datastores(path_i, frequency_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names, include_benchmark=True)    


    ctx = PlotContext(
        ds_dict=ds_dict,
        target_dict=target_dict,
        RUL_dict=RUL_dict,
        States_dict=States_dict,
        train_ds_names=train_ds_names,
        validation_ds_names=val_ds_names,
        test_ds_names=test_ds_names,
    )

    ctx_bench = PlotContext(
        ds_dict=ds_dict_bench,
        target_dict=target_dict_bench,
        RUL_dict=RUL_dict_bench,
        States_dict=States_dict_bench,
        train_ds_names=train_ds_names,
        validation_ds_names=val_ds_names,
        test_ds_names=test_ds_names,
    )

    # Load network with custom layer registration
    custom_objs = {"KSparse": KSparse}
    model = tf.keras.models.load_model(
        "C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\python_project\\results_hype_tweaked\\deep_fully_connected_autoencoder_only_proxy99.1203.h5",
        custom_objects=custom_objs,
        compile=False,
        safe_mode=False,  # allow Lambda deserialization from trusted file
    )

    model_bench = tf.keras.models.load_model(
        "C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\python_project\\results_hype_tweaked\\deep_CNN_autoencoder_only_proxy99.4161.h5",
        custom_objects=custom_objs,
        compile=False,
        safe_mode=False,  # allow Lambda deserialization from trusted file
    )


    
    state_idx = 20 # which state to visualize (0-based index), assuming that the test data has batch size 1
    plot_sHI_vs_RUL(model, ctx, state_idx, "train", save=False, dir="C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\results\\verification_plot_shi",  show=True)
    plot_sHI_vs_RUL(model, ctx, state_idx, "test", save=False, dir="C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\results\\verification_plot_shi", show=True)

    plot_sHI_vs_RUL(model_bench, ctx_bench, state_idx, "train", save=False, dir="C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\results\\verification_plot_shi",  show=True)
    plot_sHI_vs_RUL(model_bench, ctx_bench, state_idx, "test", save=False, dir="C:\\Users\\Maria\\Documents\\Honours Programme\\Networks\\GAN\\results\\verification_plot_shi", show=True)


# benchmark 
# time depencecy 
# CNN 
# results in excel file for differnet iterations of the model 
# test set 
# train with different seeds check for monotonicty  a novel machnie learning model to historical independent helath indicatior 
# Possibility of connection with Pablo 
# 