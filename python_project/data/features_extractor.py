'''
frequency domain features from vibration signals.
This code implements extraction of 19 time domain features and 14 frequency domain features.
The class inherits from the states class 
''' 
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "algorithms", "training", "viz", "scripts"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from states import states
import tensorflow as tf
import tensorflow_probability as tfp
from config import BASE_PANELS, PANEL_123_SUBPANELS, mat_file_path, FEATURES_CACHE_DIR, DEFAULT_FREQ_INDEX

class FeaturesExtractor(states):
    def __init__(self, mat_file):
        super().__init__(mat_file=mat_file)
        self.mat_file = mat_file


    # Time domain features 
    def extract_mean(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return np.mean(self.benchmark_amplitude(s_idx, freq, p_idx), axis=-1)
        return tf.reduce_mean(self.amplitude(s_idx, freq, p_idx), axis=-1)
    
    def extract_std(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return  tf.math.reduce_std(self.benchmark_amplitude(s_idx, freq, p_idx), axis=-1)
        return tf.math.reduce_std(self.amplitude(s_idx, freq, p_idx), axis=-1)
    
    def extract_root_amplitude(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return tf.pow(tf.reduce_mean(tf.sqrt(tf.abs(self.benchmark_amplitude(s_idx, freq, p_idx))), axis=-1), 2)
        return tf.pow(tf.reduce_mean(tf.sqrt(tf.abs(self.amplitude(s_idx, freq, p_idx))), axis=-1), 2)

    def extract_rms(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return tf.sqrt(tf.reduce_mean(tf.square(self.benchmark_amplitude(s_idx, freq, p_idx)), axis=-1))
        return tf.sqrt(tf.reduce_mean(tf.square(self.amplitude(s_idx, freq, p_idx)), axis=-1))
    
    def extract_rss(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return tf.sqrt(tf.reduce_sum(tf.pow(self.benchmark_amplitude(s_idx, freq, p_idx), 2), axis=-1))
        return tf.sqrt(tf.reduce_sum(tf.pow(self.amplitude(s_idx, freq, p_idx), 2), axis=-1))
    
    def extract_peak(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            return tf.reduce_max(tf.abs(self.benchmark_amplitude(s_idx, freq, p_idx)), axis=-1)
        return tf.reduce_max(tf.abs(self.amplitude(s_idx, freq, p_idx)), axis=-1)

    def extract_skweness(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)
        std = tf.math.reduce_std(amp, axis=-1, keepdims=True)
        n = tf.cast(tf.shape(amp)[-1], amp.dtype)
        numerator = tf.reduce_sum(tf.pow(amp - mean, 3), axis=-1)
        denominator = tf.squeeze(tf.pow(std, 3), axis=-1) * (n - 1)
        return numerator / denominator

    def extract_kurtosis(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)
        std  = tf.math.reduce_std(amp, axis=-1, keepdims=True)
        n    = tf.cast(tf.shape(amp)[-1], amp.dtype)
        numerator   = tf.reduce_sum(tf.pow(amp - mean, 4), axis=-1)
        denominator = tf.squeeze(tf.pow(std, 4), axis=-1) * (n - 1)
        return numerator / denominator

    def extract_crest_factor(self, s_idx, freq, p_idx, benchmark=False):
        peak = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        rms = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
        return peak / rms

    def extract_clearance_factor(self, s_idx, freq, p_idx, benchmark=False):
        peak = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        root_amplitude = self.extract_root_amplitude(s_idx, freq, p_idx, benchmark=benchmark)
        return peak / root_amplitude

    def extract_shape_factor(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            rms = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            rms = self.extract_rms(s_idx, freq, p_idx)
            amp = self.amplitude(s_idx, freq, p_idx)
        N = tf.cast(tf.shape(amp)[-1], amp.dtype)
        mean = tf.reduce_sum(np.abs(amp), axis=-1) / N
        return rms / mean

    def extract_impulse_factor(self, s_idx, freq, p_idx, benchmark=False):
        peak = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        N = tf.cast(tf.shape(amp)[-1], amp.dtype)
        mean = tf.reduce_sum(np.abs(amp), axis=-1) / N
        return peak / mean
    
    def extract_maxmin_difference(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        return tf.reduce_max(amp, axis=-1) - tf.reduce_min(amp, axis=-1)

    def exctract_k_central_moment(self, s_idx, freq, p_idx, k, benchmark=False):
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        mean = tf.reduce_mean(amp, axis=-1, keepdims=True)

        return tf.reduce_mean(tf.pow(amp - mean, k), axis=-1)

    def extract_FM4(self, s_idx, freq, p_idx, benchmark=False):
        
        k4 = self.exctract_k_central_moment(s_idx, freq, p_idx, 4, benchmark=benchmark)
        std = self.extract_std(s_idx, freq, p_idx, benchmark=benchmark)
        return k4 / tf.pow(std, 4)

    def extract_median(self, s_idx, freq, p_idx, benchmark=False):
        if benchmark:
            amp = self.benchmark_amplitude(s_idx, freq, p_idx)
        else:
            amp = self.amplitude(s_idx, freq, p_idx)
        return tfp.stats.percentile(amp, 50.0, axis=-1)


    # ------------------------------------------------------------------ #
    # Frequency domain features S1–S14  (Table A.14)                      #
    # f_k : frequency axis (Hz),  s_k : one-sided power spectrum |FFT|²   #
    # ------------------------------------------------------------------ #

    def _power_spectrum(self, s_idx, freq_idx, p_idx, sample_rate=1.0, benchmark=False):
        # s_idx may be ":" (all states → 2D) or an int (single state → 1D)
        if benchmark:
            x = self.benchmark_amplitude(s_idx, freq_idx, p_idx)
        else:
            x = self.amplitude(s_idx, freq_idx, p_idx)
        if hasattr(x, 'numpy'):
            x = x.numpy()
        x = np.asarray(x, dtype=float)
        N = x.shape[-1]                              # signal length, not n_states
        s_k = np.abs(np.fft.rfft(x, axis=-1)) ** 2  # (..., N_f)
        f_k = np.fft.rfftfreq(N, d=1.0 / sample_rate)  # (N_f,)
        return f_k, s_k

    # For all S-features:
    #   f  : (N_f,)              — frequency axis
    #   s  : (N_f,) or (B, N_f) — power spectrum, B = number of states
    # Intermediate reductions use keepdims=True so they broadcast back with f.
    # Final results are scalar or (B,).

    def extract_S1(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S1 = mean(S)"""
        _, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.mean(s, axis=-1)

    def extract_S2(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S2 = var(S)"""
        _, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.var(s, axis=-1)

    def extract_S3(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S3 = skewness(S)"""
        from scipy.stats import skew
        _, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return skew(s, axis=-1)

    def extract_S4(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S4 = kurtosis(S)"""
        from scipy.stats import kurtosis
        _, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return kurtosis(s, axis=-1)

    def extract_S5(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S5 = X_fc = Σ(f_k · s_k) / Σ(s_k)  (centroid frequency)"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.sum(f * s, axis=-1) / np.sum(s, axis=-1)

    def extract_S6(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S6 = sqrt(Σ((f_k − S5)² · s_k) / N_f)"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        return np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / len(f))

    def extract_S7(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S7 = X_rmsf = sqrt(Σ(f_k² · s_k) / Σ(s_k))"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.sqrt(np.sum(f ** 2 * s, axis=-1) / np.sum(s, axis=-1))

    def extract_S8(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S8 = sqrt(Σ(f_k⁴ · s_k) / Σ(f_k² · s_k))"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.sqrt(np.sum(f ** 4 * s, axis=-1) / np.sum(f ** 2 * s, axis=-1))

    def extract_S9(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S9 = Σ(f_k² · s_k) / sqrt(Σ(s_k) · Σ(f_k⁴ · s_k))"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        return np.sum(f ** 2 * s, axis=-1) / np.sqrt(np.sum(s, axis=-1) * np.sum(f ** 4 * s, axis=-1))

    def extract_S10(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S10 = S6 / S5  (normalised RMS spectral deviation)"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / len(f))
        return s6 / s5.squeeze(-1)

    def extract_S11(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S11 = Σ((f_k − S5)³ · s_k) / (N_f · S6³)"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum((f - s5) ** 3 * s, axis=-1) / (n * s6 ** 3)

    def extract_S12(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S12 = Σ((f_k − S5)⁴ · s_k) / (N_f · S6⁴)"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum((f - s5) ** 4 * s, axis=-1) / (n * s6 ** 4)

    def extract_S13(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S13 = Σ(sqrt(|f_k − S5|) · s_k) / (N_f · sqrt(S6))"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        n = len(f)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        s6 = np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / n)
        return np.sum(np.sqrt(np.abs(f - s5)) * s, axis=-1) / (n * np.sqrt(s6))

    def extract_S14(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False):
        """S14 = sqrt(Σ((f_k − S5)² · s_k) / Σ(s_k))"""
        f, s = self._power_spectrum(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
        s5 = np.sum(f * s, axis=-1, keepdims=True) / np.sum(s, axis=-1, keepdims=True)
        return np.sqrt(np.sum((f - s5) ** 2 * s, axis=-1) / np.sum(s, axis=-1))

    def extract_all_features(self, s_idx, freq, p_idx, sample_rate=1.0, benchmark=False, cache_dir=None):

        if p_idx == ":":
            print(f"Extracting features for all paths at freq {freq} Hz...")
            if cache_dir is not None:
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_all_paths_benchmark.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_all_paths.npy"
                if cache_file.exists():
                    return np.load(str(cache_file))
                else:
                    # extract features for all paths and save to cache
                    all_features = []

                    for p in range(0,28):
                        if benchmark:
                            features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=False, cache_dir=None)
                            features_p_ben = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=True, cache_dir=None)
                            features_p = np.concatenate([features_p, features_p_ben], axis=-1)
                        else:
                            features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=False, cache_dir=None)
                        all_features.append(features_p)
                    all_features = np.stack(all_features, axis=0)  # (n_paths, n_states, n_features)

                    # create cache directory if it doesn't exist
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    np.save(str(cache_file), all_features)
                    return all_features
            else:
                input(f"Extracting features without caching for all paths at freq {freq} Hz... Press Enter to continue.")
                all_features = []

                for p in range(0,28):
                    features_p = self.extract_all_features(s_idx, freq, p_idx=p, sample_rate=sample_rate, benchmark=benchmark, cache_dir=None)
                    all_features.append(features_p)
                all_features = np.stack(all_features, axis=0)  # (n_paths, n_states, n_features)
                return all_features         

        else:

            if cache_dir is not None:
                
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_benchmark.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}.npy"
                if cache_file.exists():
                    return np.load(str(cache_file))

            self.features = {}
            self.features['mean'] = self.extract_mean(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['std'] = self.extract_std(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['root_amplitude'] = self.extract_root_amplitude(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['rms'] = self.extract_rms(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['rss'] = self.extract_rss(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['peak'] = self.extract_peak(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['skweness'] = self.extract_skweness(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['kurtosis'] = self.extract_kurtosis(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['crest_factor'] = self.extract_crest_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['clearance_factor'] = self.extract_clearance_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['shape_factor'] = self.extract_shape_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['impulse_factor'] = self.extract_impulse_factor(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['maxmin_difference'] = self.extract_maxmin_difference(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['3rd_central_moment'] = self.exctract_k_central_moment(s_idx, freq, p_idx, 3, benchmark=benchmark)
            self.features['4th_central_moment'] = self.exctract_k_central_moment(s_idx, freq, p_idx, 4, benchmark=benchmark)
            self.features['5th_central_moment'] = self.exctract_k_central_moment(s_idx, freq, p_idx, 5, benchmark=benchmark)
            self.features['6th_central_moment'] = self.exctract_k_central_moment(s_idx, freq, p_idx, 6, benchmark=benchmark)
            self.features['FM4'] = self.extract_FM4(s_idx, freq, p_idx, benchmark=benchmark)
            self.features['median'] = self.extract_median(s_idx, freq, p_idx, benchmark=benchmark)

            # Frequency domain features
            self.features['S1'] = self.extract_S1(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S2'] = self.extract_S2(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S3'] = self.extract_S3(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S4'] = self.extract_S4(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S5'] = self.extract_S5(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S6'] = self.extract_S6(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S7'] = self.extract_S7(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S8'] = self.extract_S8(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S9'] = self.extract_S9(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S10'] = self.extract_S10(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S11'] = self.extract_S11(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S12'] = self.extract_S12(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S13'] = self.extract_S13(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)
            self.features['S14'] = self.extract_S14(s_idx, freq, p_idx, sample_rate, benchmark=benchmark)

            
            cols = [np.asarray(v) for v in self.features.values()]
            out = np.stack(cols, axis=1)  # (n_states, n_features)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            if cache_dir is not None:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                panel_name = Path(self.mat_file).stem
                if benchmark:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}_benchmark.npy"
                else:
                    cache_file = Path(cache_dir) / f"{panel_name}_freq{freq}_path{p_idx}.npy"
                np.save(str(cache_file), out)

            return out

    @staticmethod
    def normalize(train_feat, *other_feats):
        """Z-score normalize using training set statistics.
        Returns normalized train array and any additional arrays scaled by the same stats.
        Features with zero or near-zero std (constant columns) are left as zeros.
        """
        mean = np.nanmean(train_feat, axis=0)   # ignore NaN when computing stats
        std  = np.nanstd(train_feat, axis=0)
        std[std < 1e-10] = 1.0                  # catch both zero and near-zero std

        bad_cols = np.where(np.isnan(mean) | np.isnan(std))[0]
        if bad_cols.size:
            print(f"[normalize] {bad_cols.size} all-NaN columns (indices {bad_cols}) — set to 0")
            mean[bad_cols] = 0.0
            std[bad_cols]  = 1.0

        normed = [(arr - mean) / std for arr in (train_feat, *other_feats)]
        normed = [np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in normed]
        normed = [np.clip(a, -10, 10) for a in normed]
        return normed if other_feats else normed[0]
    


if __name__ == "__main__":
    for panel_name in BASE_PANELS + PANEL_123_SUBPANELS:
        mat_file = str(mat_file_path(panel_name))
        extractor = FeaturesExtractor(mat_file)
        sampling_rate = 1 / extractor.dt()
        features = extractor.extract_all_features(s_idx=":", freq=DEFAULT_FREQ_INDEX, p_idx=":", sample_rate=sampling_rate, benchmark=True, cache_dir=str(FEATURES_CACHE_DIR))
        print(f"Extracted features shape for States_{panel_name}.mat: {features.shape}")
   