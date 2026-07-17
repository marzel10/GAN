'''
Loads a saved cross-validation ensemble model and plots its sHI predictions for one panel.
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

from ae_cross_validation_helper import ClipLayer, plot_ensemble_sHI
from fc_AE import KSparse, ExpandLastDim, SqueezeLastDim
from states_check import prepare_datastores
from config import DEFAULT_FREQ_INDEX
from big_train import monotonicity_loss

model_path = "C:\\Users\\Maria\\Documents\\Honours Programme\\GAN\\python_project\\Cross_Validation_Results_AE\\path_0\\ensemble_model.keras"

# Custom layers/losses used inside the per-fold models nested in the ensemble (see
# build_ensemble_ae / build_fc_AE_features / build_CNN_AE_features) -- load_model needs
# them explicitly since none of them are registered with @keras.saving.register_keras_serializable.
# monotonicity_loss is needed even with compile=False: the nested per-fold sub-models each
# saved their own compile_config (from when they were trained/saved individually), and that
# gets deserialized regardless of the outer ensemble's own compile=False.
custom_objects = {
    "KSparse": KSparse,
    "ExpandLastDim": ExpandLastDim,
    "SqueezeLastDim": SqueezeLastDim,
    "ClipLayer": ClipLayer,
    "monotonicity_loss": monotonicity_loss,
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

path_i = 0
freq_i = DEFAULT_FREQ_INDEX
panel_number = "105"

# This saved ensemble's folds are the 4 base panels (see Model_val_<panel>.keras next to
# ensemble_model.keras on disk) -- to recover panel_number's raw features we need the
# norm_stats from the fold that held panel_number out of training (Model_val_<panel_number>),
# which is also the fold that never saw panel_number's data during training.
cv_panels = ["103", "104", "123", "109"]
train_ds_names = [p for p in cv_panels if p != panel_number]

# ref_ds_dict/norm_stats: the ensemble re-normalizes raw input per branch internally (see
# build_ensemble_ae), so plot_ensemble_sHI needs this fold's own norm_stats to first recover
# raw features from its normalized ds_dict entries.
_, _, _, ds_dict, _, _, States_dict, norm_stats = prepare_datastores(
    path_i=path_i, freq_i=freq_i, base_batch_size=8, test_batch_size=1,
    train_ds_names=train_ds_names, val_ds_names=[panel_number], test_ds_names=[panel_number],
    features=True, include_benchmark=True,
)

save_path = str(Path(model_path).parent / f"ensemble_sHI_panel_{panel_number}.png")
plot_ensemble_sHI(model, ds_dict, norm_stats, States_dict, [panel_number], save_path=save_path)
print(f"Saved: {save_path}")
