from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tensorflow as tf

class states:
    def __init__(self, mat_file):
        # Prefer scipy for v7.2 and earlier; fall back to hdf5storage for v7.3 HDF5 files
       
        data = loadmat(mat_file)
        self.file_path = mat_file
        self.states = data['States']
        # find States_ in file name and extract the part after it until the next dot or end of string
        begin_idx = mat_file.find("States_")
        if begin_idx == -1:
            raise ValueError(f"File name {mat_file} does not contain 'States_'")
        begin_idx += len("States_")
        end_idx = mat_file.find(".", begin_idx)
        if end_idx == -1:
            end_idx = len(mat_file)
        self.panel_name = mat_file[begin_idx:end_idx]
        self.num_states = self.states.shape[1]


        self.num_freq = self.states[0,0]['Frequency'].shape[1]
        self.num_pair = self.states[0,0]['Frequency'][0,0]['Pair_idx'].shape[1]

        freq_counts = [self.states[0,i]['Frequency'].shape[1] for i in range(self.num_states)]
        if len(set(freq_counts)) != 1:
            print(freq_counts)
            warnings.warn("Number of frequencies varies across states.")

        pair_counts = [self.states[0,i]['Frequency'][0,j]['Pair_idx'].shape[1] for i in range(self.num_states) for j in range(self.states[0,i]['Frequency'].shape[1])]
        if len(set(pair_counts)) != 1:
            print(pair_counts)
            warnings.warn("Number of pair indices varies across frequencies and states.")

        # Some datasets were saved with a typo in the benchmark field name.
        # Keep both here for backward/forward compatibility.
        self._benchmark_field_candidates = (
            'Benchmark_Amplitude',
            'Banchmark_Amplitude',
        )

    def _get_first_existing_field(self, struct_obj, candidate_fields):
        names = getattr(getattr(struct_obj, 'dtype', None), 'names', None)
        if not names:
            return None
        for field in candidate_fields:
            if field in names:
                return field
        return None


    def size_summary(self):
        print(f"Total states: {self.num_states}")

        freq_counts = [self.states[0,i]['Frequency'].shape[1] for i in range(self.num_states)]
        print(f"Number of frequencies per state: {freq_counts[0]} (assuming all states have the same number of frequencies)")
        if len(set(freq_counts)) != 1:
            warnings.warn("Number of frequencies varies across states.")

        pair_counts = [self.states[0,i]['Frequency'][0,j]['Pair_idx'].shape[1] for i in range(self.num_states) for j in range(self.states[0,i]['Frequency'].shape[1])]
        print(f"Number of pair indices per frequency: {pair_counts[0]} (assuming all frequencies have the same number of pair indices)")
        if len(set(pair_counts)) != 1:
            warnings.warn("Number of pair indices varies across frequencies and states.")

    def amplitude(self, s_idx, freq, p_idx):
        # Support broadcasting over states via None/"all"/slice to mirror MATLAB's ':' intent
        if freq >= self.num_freq or p_idx >= self.num_pair:
            raise ValueError(f"Index out of bounds. Max frequency index: {self.num_freq-1}, max pair index: {self.num_pair-1}")

        if s_idx in (None, ":", "all"):
            state_indices = range(self.num_states)
        elif isinstance(s_idx, slice):
            state_indices = range(self.num_states)[s_idx]
        elif isinstance(s_idx, (list, tuple, np.ndarray)):
            state_indices = s_idx
        else:
            if s_idx >= self.num_states:
                raise ValueError(f"Index out of bounds. Max state index: {self.num_states-1}")
            return self.states[0,s_idx]['Frequency'][0,freq]['Pair_idx'][0,p_idx]['Amplitude']

        amps = []
        for idx in state_indices:
            if idx >= self.num_states:
                raise ValueError(f"Index out of bounds. Max state index: {self.num_states-1}")
            amps.append(tf.squeeze(self.states[0,idx]['Frequency'][0,freq]['Pair_idx'][0,p_idx]['Amplitude']))

        try:
            return np.stack(amps, axis=0)
        except ValueError:
            # Fallback if lengths differ; return a list for manual handling
            return amps
    
    def signal_energy(self, s_idx, freq, p_idx):
        amp = self.amplitude(s_idx, freq, p_idx)
        dim = np.where(np.array(amp.shape) == 4000)[0]
        return tf.reduce_sum(tf.square(amp), axis=dim)
    
    def benchmark_amplitude(self, s_idx, freq, p_idx):
        if freq >= self.num_freq or p_idx >= self.num_pair:
            raise ValueError(
                f"Index out of bounds. Max frequency index: {self.num_freq-1}, max pair index: {self.num_pair-1}"
            )
        
        if s_idx in (None, ":", "all"):
            state_indices = range(self.num_states)
        elif isinstance(s_idx, slice):
            state_indices = range(self.num_states)[s_idx]
        elif isinstance(s_idx, (list, tuple, np.ndarray)):
            state_indices = s_idx
        else:
            if s_idx >= self.num_states:
                raise ValueError(f"Index out of bounds. Max state index: {self.num_states-1}")
            pair_struct = self.states[0,s_idx]['Frequency'][0,freq]['Pair_idx'][0,p_idx]
            bench_field = self._get_first_existing_field(pair_struct, self._benchmark_field_candidates)
            if not bench_field:
                warnings.warn(
                    f"Benchmark field not found for state {s_idx}, frequency {freq}, pair index {p_idx}. Returning zeros."
                )
                return tf.reshape(tf.squeeze(tf.zeros_like(self.amplitude(s_idx, freq, p_idx))), [-1])

            bench = pair_struct[bench_field]
            return tf.reshape(tf.squeeze(tf.convert_to_tensor(bench)), [-1])
        
        bench = []
        for idx in state_indices:
            if idx >= self.num_states:
                raise ValueError(f"Index out of bounds. Max state index: {self.num_states-1}")
            
            pair_struct = self.states[0,idx]['Frequency'][0,freq]['Pair_idx'][0,p_idx]
            bench_field = self._get_first_existing_field(pair_struct, self._benchmark_field_candidates)
            if not bench_field:
                warnings.warn(
                    f"Benchmark field not found for state {idx}, frequency {freq}, pair index {p_idx}. Returning zeros."
                )
                bench.append(tf.squeeze(tf.zeros_like(self.amplitude(idx, freq, p_idx))))
            else:
                bench.append(tf.squeeze(pair_struct[bench_field]))

        return tf.stack(bench, axis=0)
    
    def time(self, s_idx, freq, p_idx):
        if s_idx >= self.num_states or freq >= self.num_freq or p_idx >= self.num_pair:
            raise ValueError(f"Index out of bounds. Max state index: {self.num_states-1}, max frequency index: {self.num_freq-1}, max pair index: {self.num_pair-1}")
        return tf.squeeze(self.states[0,s_idx]['Frequency'][0,freq]['Pair_idx'][0,p_idx]['Time'])
    
    def dt(self):
        return self.states[0,0]['Frequency'][0,0]['Pair_idx'][0,0]['Time'][1] - self.states[0,0]['Frequency'][0,0]['Pair_idx'][0,0]['Time'][0]

    def plot(self, s_idx, freq, p_idx):
        amp = self.amplitude(s_idx, freq, p_idx)
        benchmark_amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        time = self.time(s_idx, freq, p_idx)

        plt.plot(time, amp, label='Amplitude')
        plt.plot(time, benchmark_amp, label='Benchmark Amplitude')
        plt.title(f'State {s_idx}, Frequency {freq}, Pair Index {p_idx}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
