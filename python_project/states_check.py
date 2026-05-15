import sys
from pathlib import Path

# Allow running this file directly (e.g., `python python_project/states_check.py`)
# by ensuring the repository root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python_project.states import states
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


_DATA_DIR = Path(__file__).resolve().parent

st_103 = states(str(_DATA_DIR / "data/States_103.mat"))
st_104 = states(str(_DATA_DIR / "data/States_104.mat"))
st_105 = states(str(_DATA_DIR / "data/States_105.mat"))
st_109 = states(str(_DATA_DIR / "data/States_109.mat"))
st_123_1 = states(str(_DATA_DIR / "data/States_123_1.mat"))
st_123_2 = states(str(_DATA_DIR / "data/States_123_2.mat"))
st_123_31 = states(str(_DATA_DIR / "data/States_123_31.mat"))
st_123_32 = states(str(_DATA_DIR / "data/States_123_32.mat"))
st_123_41 = states(str(_DATA_DIR / "data/States_123_41.mat"))
st_123_42 = states(str(_DATA_DIR / "data/States_123_42.mat"))
st_123_43 = states(str(_DATA_DIR / "data/States_123_43.mat"))
st_123_44 = states(str(_DATA_DIR / "data/States_123_44.mat"))

states_list = [st_103, st_104, st_105, st_109, st_123_1, st_123_2, st_123_31, st_123_32, st_123_41, st_123_42, st_123_43, st_123_44]
ds_names = ["103", "104", "105", "109", "123_1", "123_2", "123_31", "123_32", "123_41", "123_42", "123_43", "123_44"]
    

def prepare_simple_dataset(path_i, freq_i, panel_name,batch_size=1,include_benchmark=False):
    if freq_i >= states_list[0].num_freq or path_i >= states_list[0].num_pair:
        raise ValueError(f"Index out of bounds. Max frequency index: {states_list[0].num_freq-1}, max pair index: {states_list[0].num_pair-1}")
    if panel_name.startswith("123"):
        # For panel 123, we concatenate all states from the 4 sub-panels
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
        st = np.where(np.array(ds_names) == panel_name)[0]
        st = states_list[st[0]]
        amp = st.amplitude(":", freq_i, path_i)
        if include_benchmark:
            bench = st.benchmark_amplitude(":", freq_i, path_i)
            ds, _ = make_ds(amp, st.num_states, batch_size=batch_size, benchmark=bench)
        else:
            ds, _ = make_ds(amp, st.num_states, batch_size=batch_size)
        return ds

def prepare_datastores(path_i, freq_i, base_batch_size, test_batch_size, train_ds_names, val_ds_names, test_ds_names, include_benchmark=False):
    if freq_i >= states_list[0].num_freq or path_i >= states_list[0].num_pair:
        raise ValueError(f"Index out of bounds. Max frequency index: {states_list[0].num_freq-1}, max pair index: {states_list[0].num_pair-1}")
    total_states_123 = sum(st.num_states for st in states_list[4:])
    starting_states = np.cumsum([0] + [st.num_states for st in states_list[4:]])
    amp_list = []
    if include_benchmark:
        bench_list = []

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

    return train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict




if __name__ == "__main__":

    freq = 3
    path_i = 0
    base_batch_size = 30
    test_batch_size = 1

    # Training datasets
    train_ds_names = ["103", "104", "105", "109"]
    val_ds_names = ["123_1", "123_31", "123_41", "123_43"] 
    test_ds_names = ["123_2", "123_32", "123_42", "123_44"]

    include_benchmark = True
    train_dataset, val_dataset, test_dataset, ds_dict, target_dict, RUL_dict, States_dict = prepare_datastores(
        path_i,
        freq,
        base_batch_size,
        test_batch_size,
        train_ds_names,
        val_ds_names,
        test_ds_names,
        include_benchmark=include_benchmark,
    )

    recon_key = "final_1" if include_benchmark else "fc_output_1"

    # Example: iterate one batch
    for x, y in train_dataset.take(1):
        print("batch shape", x.shape)
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
