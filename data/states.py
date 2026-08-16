'''
This file defines the states class for loading and accessing the .mat files containing the panel states. 
It provides methods to extract:
 - amplitude
 - benchmark amplitude
 - time
 - signal energy 

Other functions include:
    - size_summary: prints the number of states, frequencies, and pair indices in the loaded panel
    - plot: plots the amplitude and benchmark amplitude over time
    - dt: calculates the time step between samples based on the time vector in the .mat file

Example of usage is in the states_check.py 
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

from scipy.io import loadmat
from scipy.signal import hilbert
import h5py
import matplotlib.pyplot as plt
import numpy as np

from config import mat_file_path
import warnings
import tensorflow as tf

class states:
    def __init__(self, mat_file):
        # Prefer scipy for v7.2 and earlier; fall back to hdf5storage for v7.3 HDF5 files
       
        data = loadmat(mat_file)
        self.file_path = mat_file
        self.states = data['States']
        # find States_ in the file's stem (name without extension) and extract the part after it
        stem = Path(mat_file).stem
        begin_idx = stem.find("States_")
        if begin_idx == -1:
            raise ValueError(f"File name {mat_file} does not contain 'States_'")
        begin_idx += len("States_")
        self.panel_name = stem[begin_idx:]
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

    def signal_envelope_peak(self, s_idx, freq, p_idx):
        amp = np.asarray(self.amplitude(s_idx, freq, p_idx)).reshape(-1)
        envelope = np.abs(hilbert(amp))
        return tf.reduce_max(envelope)

    def signal_envelope_peak_area(self, s_idx, freq, p_idx):
        amp = np.asarray(self.amplitude(s_idx, freq, p_idx)).reshape(-1)
        envelope = np.abs(hilbert(amp))
        peak_idx = int(np.argmax(envelope))

        area = 0.0
        i = peak_idx
        while i > 0 and envelope[i - 1] <= envelope[i]:
            area += 0.5 * (envelope[i] + envelope[i - 1])
            i -= 1

        i = peak_idx
        while i < len(envelope) - 1 and envelope[i + 1] <= envelope[i]:
            area += 0.5 * (envelope[i] + envelope[i + 1])
            i += 1

        return area * self.dt()


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

    def plot(self, s_idx, freq, p_idx, save_path=None):
        amp = self.amplitude(s_idx, freq, p_idx)
        benchmark_amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        time = self.time(s_idx, freq, p_idx)

        #square figure
        plt.figure(figsize=(6, 6))
        plt.plot(time[2100:3000], amp[2100:3000], label='Amplitude')
        plt.plot(time[2100:3000], benchmark_amp[2100:3000], label='Benchmark Amplitude')
        plt.title(f'State {s_idx}, Frequency {freq}, Pair Index {p_idx}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
        if save_path is not None:
            plt.savefig(save_path, format='svg', bbox_inches='tight')
       

