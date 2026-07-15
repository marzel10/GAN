'''
This file:
- defines the prepare_simple_dataset function for creating TensorFlow datasets from the states data (simple input to the AE)
- defines the prepare_datastores function for creating train/val/test datasets with RUL targets and optional benchmark features
- includes a main block that demonstrates how to use these functions to prepare datasets and plot some example data and RUL curves.

'''
from ast import If
import sys
from pathlib import Path
import warnings

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from states import states
from features_extractor import FeaturesExtractor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from config import (
    BASE_PANELS, PANEL_123_SUBPANELS, VAL_123_SUBPANELS, TEST_123_SUBPANELS,
    mat_file_path, DEFAULT_FREQ_INDEX,
)


def compute_weibull_health(N):
    min_stress        = -6.5
    max_stress        = -65
    ultimate_strength = -104

    amp_stress      = abs(max_stress - min_stress) / 2
    mean_stress     = (max_stress + min_stress) / 2
    nor_amp_stress  = amp_stress  / ultimate_strength
    nor_mean_stress = mean_stress / ultimate_strength

    k = -1.17 * nor_mean_stress - 19.9 * nor_amp_stress + 5.92
    lam = 0.0661 * nor_mean_stress - 1.05 * nor_amp_stress + 0.754

    weibull_health = np.zeros(N, dtype=float)

    for i in range(N):
        x = (i + 1) / N
        val = lam * (np.log(x) + 0j) ** (1 / k)   # allow complex intermediate
        weibull_health[i] = np.real(val)

    return tf.convert_to_tensor(weibull_health, dtype=tf.float32)

# Build one dataset per panel (handles different state counts), then concatenate
def make_ds(arr, N, batch_size, state_idx=None, benchmark=None):
    arr = tf.convert_to_tensor(arr, dtype=tf.float32)
    if benchmark is not None:
        benchmark = tf.convert_to_tensor(benchmark)
        benchmark = tf.cast(tf.squeeze(benchmark), tf.float32)
        # Arrange as (time, features) per sample: (4000, 2)
        # (Using axis=1 would give (2, 4000), which is why you saw (batch, 2, 4000).)
        arr = tf.stack([arr, benchmark], axis=-1)
    weibul = compute_weibull_health(N)
    if state_idx is not None:
        idx = tf.convert_to_tensor(state_idx, dtype=tf.int32)
        weibul = tf.gather(weibul, idx)
    x_ds = tf.data.Dataset.from_tensor_slices(arr)
    w_ds = tf.data.Dataset.from_tensor_slices(weibul)
    
    # Pair each sample with its scalar health label
    ds = tf.data.Dataset.zip((x_ds, w_ds))

    def _pack(x, w):
        # Target dict keys must match the model output layer names.
        # - Fully-connected AE reconstruction output: "fc_output_1"
        # - CNN AE reconstruction output (after crop/pad): "final_1"
        recon_key = "final_1" if benchmark is not None else "fc_output_1"
        return x, {recon_key: x, "fc_latent_1": w}

    return ds.map(_pack).batch(batch_size, drop_remainder=False), weibul.numpy()

def make_ds_features(arr, N, batch_size, state_idx=None):
    arr = tf.convert_to_tensor(arr, dtype=tf.float32)
    weibul = compute_weibull_health(N)
    if state_idx is not None:
        idx = tf.convert_to_tensor(state_idx, dtype=tf.int32)
        weibul = tf.gather(weibul, idx)
    x_ds = tf.data.Dataset.from_tensor_slices(arr)
    w_ds = tf.data.Dataset.from_tensor_slices(weibul)
    ds = tf.data.Dataset.zip((x_ds, w_ds))

    def _pack(x, w):
        # Output names must match build_fc_AE_features: "reconstruction" and "sHI"
        return x, {"reconstruction": x, "sHI": w}

    return ds.map(_pack).batch(batch_size, drop_remainder=False), weibul.numpy()


ds_names = BASE_PANELS + PANEL_123_SUBPANELS

states_list = [states(str(mat_file_path(name))) for name in ds_names]
features_list = [FeaturesExtractor(str(mat_file_path(name))) for name in ds_names]

st_103, st_104, st_105, st_109, st_123_1, st_123_2, st_123_31, st_123_32, st_123_41, st_123_42, st_123_43, st_123_44 = states_list
f_103, f_104, f_105, f_109, f_123_1, f_123_2, f_123_31, f_123_32, f_123_41, f_123_42, f_123_43, f_123_44 = features_list
sampling_rate = 1 / st_103.dt()
    

def prepare_simple_dataset(path_i, freq_i, panel_name,batch_size=1,include_benchmark=False, features=False):
    if freq_i >= states_list[0].num_freq or path_i >= states_list[0].num_pair:
        raise ValueError(f"Index out of bounds. Max frequency index: {states_list[0].num_freq-1}, max pair index: {states_list[0].num_pair-1}")
    if panel_name.startswith("123"):
        # For panel 123, we concatenate all states from the 4 sub-panels
        if features:
            f_idx = np.where(np.array(ds_names) == "123_1")[0]
            for f in features_list[f_idx[0]:f_idx[0]+8]:
                feat = f.extract_all_features(":", freq_i, path_i, sample_rate=sampling_rate)
                if 'full_feat' not in locals():
                    full_feat = feat
                else:
                    full_feat = np.concatenate([full_feat, feat], axis=0)
            ds, _ = make_ds_features(full_feat, full_feat.shape[0], batch_size=batch_size)
            return ds
        else:
            st_idx = np.where(np.array(ds_names) == "123_1")[0]
            for st in states_list[st_idx[0]:st_idx[0]+8]:
                if include_benchmark:
                    bench = st.benchmark_amplitude(":", freq_i, path_i)
                    amp = st.amplitude(":", freq_i, path_i)
                    if 'full_amp' not in locals():
                        full_amp = amp
                        full_bench = bench
                    else:
                        full_amp = np.concatenate([full_amp, amp], axis=0)
                        full_bench = np.concatenate([full_bench, bench], axis=0)
            
            if include_benchmark:
                ds, _ = make_ds(full_amp, full_amp.shape[0], batch_size=batch_size, benchmark=full_bench)
            else:
                ds, _ = make_ds(full_amp, full_amp.shape[0], batch_size=batch_size)
            
            return ds
    # For panels 103, 104, 105, 109, we just use the single corresponding state file.
    else:

        if features:
            f_idx = np.where(np.array(ds_names) == panel_name)[0]
            f = features_list[f_idx[0]]
            feat = f.extract_all_features(":", freq_i, path_i, sample_rate=sampling_rate)
            ds, _ = make_ds_features(feat, feat.shape[0], batch_size=batch_size)
            return ds
        else:
            st = np.where(np.array(ds_names) == panel_name)[0]
            st = states_list[st[0]]
            amp = st.amplitude(":", freq_i, path_i)
            if include_benchmark:
                bench = st.benchmark_amplitude(":", freq_i, path_i)
                ds, _ = make_ds(amp, st.num_states, batch_size=batch_size, benchmark=bench)
            else:
                ds, _ = make_ds(amp, st.num_states, batch_size=batch_size)
            return ds

def prepare_datastores(path_i, freq_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names, include_benchmark=False, features=False, diff_bench=False):
    if freq_i >= states_list[0].num_freq or path_i >= states_list[0].num_pair:
        raise ValueError(f"Index out of bounds. Max frequency index: {states_list[0].num_freq-1}, max pair index: {states_list[0].num_pair-1}")
    
    total_states_123 = sum(st.num_states for st in states_list[4:])
    starting_states = np.cumsum([0] + [st.num_states for st in states_list[4:]])
    amp_list = []
    f_list = []
    if include_benchmark:
        bench_list = []

    if features:
        for f in features_list:
            feat = f.extract_all_features(":", freq_i, path_i, sample_rate=sampling_rate, benchmark=False)
            if include_benchmark:
                feat_bench = f.extract_all_features(":", freq_i, path_i, sample_rate=sampling_rate, benchmark=True)
                if diff_bench:
                    # The feature is the difference between the signal and benchmark features
                    feat = feat - feat_bench
                else:
                    # stack next to features to create(n_states, num_features, 2)
                    feat = np.stack([feat, feat_bench], axis=-1)
            f_list.append(feat)
    else:
        for st in states_list:
            amp = st.amplitude(":", freq_i, path_i)
            
            #print(f"Amplitude shape for {st.file_path}: {amp.shape}")
            amp_list.append(amp)

            if include_benchmark:
                bench = st.benchmark_amplitude(":", freq_i, path_i)
                #print(f"Benchmark amplitude shape for {st.file_path}: {bench.shape}")
                bench_list.append(bench)
    # Build datasets and RUL arrays for each panel
    ds_dict = {}
    target_dict = {}
    RUL_dict = {}
    States_dict = {}

    if features:
        for i, f in enumerate(features_list):
            ds_name = ds_names[i]
            current_batch_size = test_batch_size if ds_name in test_ds_names else base_batch_size
            
            if ds_names[i].startswith("123"):
                idx_123 = i-4
                ds, tg = make_ds_features(f_list[i], total_states_123, batch_size=current_batch_size, state_idx=np.arange(starting_states[idx_123], starting_states[idx_123+1]))
                state_idx = np.arange(starting_states[idx_123], starting_states[idx_123+1])
                RUL_array = np.arange(total_states_123-starting_states[idx_123], total_states_123 - starting_states[idx_123 + 1], -1) / total_states_123
                RUL_dict[ds_names[i]] = RUL_array
            else:
                ds, tg = make_ds_features(f_list[i], f_list[i].shape[0], batch_size=current_batch_size)
                state_idx = np.arange(states_list[i].num_states)
                RUL_dict[ds_names[i]] = np.arange(states_list[i].num_states, 0, -1) / states_list[i].num_states
            ds_dict[ds_name] = ds
            target_dict[ds_name] = tg
            States_dict[ds_name] = state_idx
    else:
        for i, st in enumerate(states_list):
            ds_name = ds_names[i]
            current_batch_size = test_batch_size if ds_name in test_ds_names else base_batch_size
            
            if ds_names[i].startswith("123"):
                idx_123 = i-4
                if include_benchmark:
                    ds, tg = make_ds(amp_list[i], total_states_123, batch_size=current_batch_size, state_idx=np.arange(starting_states[idx_123], starting_states[idx_123+1]), benchmark=bench_list[i])
                else:
                    ds, tg = make_ds(amp_list[i], total_states_123, batch_size=current_batch_size, state_idx=np.arange(starting_states[idx_123], starting_states[idx_123+1]))
                # print(f"start {starting_states[idx_123]}, end {starting_states[idx_123+1]}, total {total_states_123}")
                RUL_array = np.arange(total_states_123-starting_states[idx_123], total_states_123 - starting_states[idx_123 + 1], -1) / total_states_123
                RUL_dict[ds_names[i]] = RUL_array
                state_idx = np.arange(starting_states[idx_123], starting_states[idx_123+1])
            else:
                RUL_dict[ds_names[i]] = np.arange(st.num_states, 0, -1) / st.num_states
                if include_benchmark:
                    ds, tg = make_ds(amp_list[i], st.num_states, batch_size=current_batch_size, benchmark=bench_list[i])
                else:
                    ds, tg = make_ds(amp_list[i], st.num_states, batch_size=current_batch_size)
                state_idx = np.arange(st.num_states)

            ds_dict[ds_names[i]] = ds
            target_dict[ds_names[i]] = tg
            States_dict[ds_names[i]] = state_idx


    # Concatenate datasets for training, validation, and testing
    for i, name in enumerate(train_ds_names):
        if i ==0:
            train_dataset = ds_dict[name]
        else:
            train_dataset = train_dataset.concatenate(ds_dict[name])
        
        if i == len(train_ds_names) - 1:
            train_dataset = train_dataset.shuffle(1000, reshuffle_each_iteration=True)

    for i, name in enumerate(val_ds_names):
        if i ==0:
            val_dataset = ds_dict[name]
        else:
            val_dataset = val_dataset.concatenate(ds_dict[name])

    for i, name in enumerate(test_ds_names):
        if i ==0:
            test_dataset = ds_dict[name]
        else:
            test_dataset = test_dataset.concatenate(ds_dict[name])

    #if features are enabled, normalize them across the entire training set (fit scaler on train, apply to val/test)
    norm_stats = None
    if features:
        # Collect all training batches then compute stats with numpy — avoids the
        # catastrophic cancellation in the one-pass formula sqrt(E[X²] - E[X]²)
        # which goes NaN when feature values are large (e.g. spectral variance S2).
        all_x = np.concatenate([x.numpy() for x, _ in train_dataset], axis=0)  # (N, 33)

        feature_means = np.nanmean(all_x, axis=0).astype(np.float32)
        feature_stds  = np.nanstd(all_x,  axis=0).astype(np.float32)

        bad = (feature_stds < 1e-10) | np.isnan(feature_stds)
        if np.any(bad):
            print(f"Warning: {bad.sum()} features have zero/NaN std at indices {np.where(bad)[0]} — those columns will be zeroed after mean subtraction.")
            feature_means[np.isnan(feature_means)] = 0.0
            feature_stds[bad] = 1.0

        def normalize(x, y):
            x_norm = (x - feature_means) / feature_stds
            x_norm = tf.clip_by_value(x_norm, -10.0, 10.0)
            return x_norm, {"reconstruction": x_norm, "sHI": y["sHI"]}

        # Apply normalization to concatenated datasets and every individual ds_dict entry
        # (ds_dict is used by plotting/inference code and must also see normalized inputs)
        train_dataset = train_dataset.map(normalize)
        val_dataset   = val_dataset.map(normalize)
        test_dataset  = test_dataset.map(normalize)
        for name in ds_dict:
            ds_dict[name] = ds_dict[name].map(normalize)

        # Stored so callers (e.g. cross-validation ensembling) can reproduce the exact
        # normalization this fold's model was trained on, and invert it back to raw units.
        norm_stats = {"feature_means": feature_means, "feature_stds": feature_stds}

    return train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict, norm_stats




if __name__ == "__main__":

    freq = DEFAULT_FREQ_INDEX
    path_i = 0
    base_batch_size = 30
    test_batch_size = 1

    # Training datasets
    train_ds_names = BASE_PANELS
    val_ds_names = VAL_123_SUBPANELS
    test_ds_names = TEST_123_SUBPANELS

    include_benchmark = True
    features = True


    train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict, norm_stats = prepare_datastores(
        path_i,
        freq,
        base_batch_size,
        test_batch_size,
        train_ds_names,
        val_ds_names,
        test_ds_names,
        include_benchmark=include_benchmark,
        features=features
    )

    recon_key = "final_1" if include_benchmark else "fc_output_1"

    if features:
        
        # Example: plot few first samples from the training dataset
        for x, y in train_dataset.take(1):
            print("batch shape", x.shape)
            print("y shape", y["reconstruction"].shape)
            print("model output shape", y["sHI"].shape)
            #max accross the states (rows)
            max_features = np.max(x,axis=1)
            min_features = np.min(x,axis=1)
            print("max features", max_features, "mean max features", np.mean(max_features))
            print("min features", min_features, "mean min features", np.mean(min_features))

            #mean value of every feature across the states (rows)
            mean_features = np.mean(x,axis=0)
            print("mean features", mean_features,mean_features.shape)
            
            # check if target is the same as the input (it should be for the AE reconstruction target)
            if np.all(x.numpy() == y["reconstruction"].numpy()):
                print("Target matches input as expected for AE reconstruction.")
            else:
                print("Warning: Target does not match input for AE reconstruction.")
            
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            for i in range(1, int(x.shape[0]*0.25)):
                axs[0].plot(x[i].numpy(), 'o', label=f'Features {i}')
                axs[1].plot(y["reconstruction"][i].numpy(), 'x', label=f'Target {i}')
            #plt.plot(y[recon_key][0].numpy(), 'x', color='red', label='Target Amplitude')
            plt.legend()
            plt.show()
            break
    else:
        for x, y in train_dataset.take(1):
            print("batch shape", x.shape)
            print("x shape", x[0].shape)
            print("y shape", y[recon_key].shape)
            print("model output shape", y["fc_latent_1"].shape)
            plt.plot(x[0].numpy(), 'o', color='blue', label='Input Amplitude')
            plt.plot(y[recon_key][0].numpy(), 'x', color='red', label='Target Amplitude')
            plt.legend()
            plt.show()
            break

    # Check how long the dataset is
    count = 0
    for _ in train_dataset:
        count += 1
    print("Dataset length:", count)
    expected_length = int(
        np.ceil(st_103.num_states / base_batch_size)
        + np.ceil(st_104.num_states / base_batch_size)
        + np.ceil(st_105.num_states / base_batch_size)
        + np.ceil(st_109.num_states / base_batch_size)
    )

    print("Expected length:", expected_length)


    # Plot RUL for the panel 123 
    plt.figure(figsize=(10, 6))
    colors = ['orange', 'green', 'blue', 'red', 'purple', 'cyan', 'magenta', 'brown']
    starting_states = [States_dict[name][0] for name in ds_names[4:]] 

    for i, name in enumerate(ds_names[4:]):
                 
        state_idx = States_dict[name]
        plt.plot(state_idx, RUL_dict[name], color=colors[i], label=name)
    plt.title('RUL Curves for Panel 123')
    plt.xlabel('State Index')
    plt.ylabel('RUL (normalized)')
    plt.legend()
    plt.show()

    #plot targets for the panel 123
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(ds_names[4:]):
        state_idx = States_dict[name]
        t = target_dict[name]
        plt.plot(state_idx, t, color=colors[i], label=name)
    plt.title('Target Amplitudes for Panel 123')
    plt.xlabel('State Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
