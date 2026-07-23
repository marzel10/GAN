'''
One-off maintenance script: regenerates every ensemble_model.keras under
Multi_path_BO_fixed_freq{0..5}/Bayesian_CNN_AE_path{0..27} using the current (fixed)
build_ensemble_ae, which now includes a working latent_space output. The existing files
on disk predate that output entirely, so extract_shi.py's
model.get_layer("latent_space") fails on them.

No retraining needed -- reloads the 4 already-trained Model_val_<panel>.keras fold
models per path/frequency, recomputes norm_stats the same deterministic way
model_train_features does (see states_check.prepare_datastores: feature_means/stds are
computed from the raw training data, independent of batch_size), rebuilds the ensemble,
and overwrites ensemble_model.keras in place.
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

import tensorflow as tf

from fc_AE import KSparse, ExpandLastDim, SqueezeLastDim
from ae_cross_validation_helper import ClipLayer, _CompatHeNormal, build_ensemble_ae
from big_train import monotonicity_loss
from states_check import prepare_datastores
from config import BASE_PANELS

CUSTOM_OBJECTS = {
    "KSparse": KSparse,
    "ExpandLastDim": ExpandLastDim,
    "SqueezeLastDim": SqueezeLastDim,
    "ClipLayer": ClipLayer,
    "monotonicity_loss": monotonicity_loss,
    "HeNormal": _CompatHeNormal,
}

N_PATHS = 28
N_FREQS = 6


def rebuild_one(freq_idx, path_i):
    path_dir = _PROJECT_ROOT / f"Multi_path_BO_fixed_freq{freq_idx}" / f"Bayesian_CNN_AE_path{path_i}"
    if not path_dir.exists():
        print(f"[freq{freq_idx}] path {path_i}: dir missing, skipping")
        return False

    fold_entries = []
    for held_out in BASE_PANELS:
        model_path = path_dir / f"Model_val_{held_out}.keras"
        if not model_path.exists():
            print(f"[freq{freq_idx}] path {path_i}: {model_path.name} missing, skipping")
            return False

        tf.keras.backend.clear_session()
        # safe_mode=False is required for custom_objects to actually override a
        # recognized built-in class name like "HeNormal" -- without it Keras 3
        # resolves HeNormal from its own keras.initializers module directly, ignoring
        # _CompatHeNormal, and older-format models fail with "unexpected keyword
        # argument 'input_axes'".
        fold_model = tf.keras.models.load_model(str(model_path), custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)

        train_ds_names = [p for p in BASE_PANELS if p != held_out]
        _, _, _, _, *_rest = prepare_datastores(
            path_i=path_i, freq_i=freq_idx, base_batch_size=16, test_batch_size=1,
            train_ds_names=train_ds_names, val_ds_names=[held_out], test_ds_names=["123"],
            include_benchmark=True, features=True,
        )
        norm_stats = _rest[-1]
        fold_entries.append((held_out, fold_model, norm_stats))

    ensemble_model = build_ensemble_ae(fold_entries)
    layer_names = [l.name for l in ensemble_model.layers]
    assert "latent_space" in layer_names, f"latent_space missing after rebuild: {layer_names}"

    ensemble_model.save(str(path_dir / "ensemble_model.keras"))
    print(f"[freq{freq_idx}] path {path_i}: ensemble rebuilt OK")
    tf.keras.backend.clear_session()
    return True


if __name__ == "__main__":
    rebuilt, skipped, failed = 0, 0, 0
    freq_to_rebuild = [0]
    for freq_idx in freq_to_rebuild:
        for path_i in range(N_PATHS):
            try:
                if rebuild_one(freq_idx, path_i):
                    rebuilt += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"[freq{freq_idx}] path {path_i}: FAILED ({e!r})")
                failed += 1
            finally:
                tf.keras.backend.clear_session()
    print(f"\nDone. Rebuilt {rebuilt}, skipped {skipped}, failed {failed} (out of {N_FREQS * N_PATHS}).")
