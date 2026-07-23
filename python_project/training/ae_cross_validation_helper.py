'''
Plotting and ensembling helpers for the autoencoder cross-validation loop in big_train.py.

Each cross-validation fold holds one of the 4 base panels out of training and produces its
own Keras model (outputs named 'sHI' and 'reconstruction', see fc_AE.build_fc_AE_features /
build_CNN_AE_features) plus its own feature normalization stats (fit only on that fold's 3
training panels, see states_check.prepare_datastores).
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
import tensorflow as tf
import matplotlib.pyplot as plt


class ClipLayer(tf.keras.layers.Layer):
    '''Elementwise tf.clip_by_value as a proper Layer (not a Lambda) so the ensemble
    model saves/reloads without Keras's Lambda-deserialization restrictions.'''

    def __init__(self, min_value=-10.0, max_value=10.0, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


'''Some fold models were saved by an older Keras version whose layer configs include
fields the currently installed Keras's __init__ signatures no longer accept (e.g.
HeNormal's input_axes/output_axes, BatchNormalization's
renorm/renorm_clipping/renorm_momentum). A custom_objects={"HeNormal": ...}
substitution can't fix this: Keras 3's custom_objects lookup keys strictly on the saved
`registered_name` field, and these older files predate
@register_keras_serializable() entirely, so registered_name is None -- Keras never even
checks custom_objects, it resolves the class straight from the real keras module. The
only place that can actually intercept this is the real class itself, so patch it in
place and drop the specific kwargs its __init__ no longer recognizes.'''


def _drop_kwargs(cls, *bad_keys):
    real_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        for key in bad_keys:
            kwargs.pop(key, None)
        real_init(self, *args, **kwargs)

    cls.__init__ = patched_init


_drop_kwargs(tf.keras.initializers.HeNormal, "input_axes", "output_axes")
_drop_kwargs(tf.keras.layers.BatchNormalization, "renorm", "renorm_clipping", "renorm_momentum")
_CompatHeNormal = tf.keras.initializers.HeNormal  # kept importable for existing CUSTOM_OBJECTS dicts


def _predict_dataset(model, ds):
    '''Runs model.predict over every batch of a datastore, returns concatenated (sHI, reconstruction) arrays.'''
    shi_idx = model.output_names.index('sHI')
    rec_idx = model.output_names.index('reconstruction')
    all_shi, all_rec = [], []
    for x, _ in ds:
        pred = model.predict(x, verbose=0)
        all_shi.append(np.asarray(pred[shi_idx]).reshape(x.shape[0], -1))
        all_rec.append(np.asarray(pred[rec_idx]))
    return np.concatenate(all_shi, axis=0), np.concatenate(all_rec, axis=0)


def plot_sHI_cv_fold(fold_model, ds_dict, States_dict, panels, held_out_panel, save_path):
    '''One figure, 4 subplots (one per base panel): sHI vs state predicted by a single
    cross-validation fold's model -- the fold that held `held_out_panel` out of training.'''
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        sHI, _ = _predict_dataset(fold_model, ds_dict[panel])
        states = States_dict[panel]
        axs[i].scatter(states, sHI[:, 0], alpha=0.7, s=15,
                        color='orange' if panel == held_out_panel else 'steelblue')
        axs[i].set_title(f'Panel {panel}' + (' (held out)' if panel == held_out_panel else ''))
        axs[i].set_xlabel('State idx')
        axs[i].set_ylabel('sHI')
        axs[i].grid(True)
    fig.suptitle(f'sHI vs State — fold validated on {held_out_panel}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_reconstruction_cv_fold(fold_model, ds_dict, States_dict, panels, held_out_panel, state_idx, save_path):
    '''One figure, 4 subplots (one per base panel): example input-vs-reconstructed signal
    at a given state, from a single cross-validation fold's model.'''
    rec_idx = fold_model.output_names.index('reconstruction')
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        states = States_dict[panel]
        offset = 0
        for x, _ in ds_dict[panel]:
            batch_size = x.shape[0]
            batch_states = states[offset:offset + batch_size]
            if np.any(batch_states == state_idx):
                local_idx = int(np.where(batch_states == state_idx)[0][0])
                pred = fold_model.predict(x, verbose=0)
                rec = np.asarray(pred[rec_idx])
                x_arr = np.asarray(x)
                axs[i].plot(x_arr[local_idx].reshape(-1), color='blue', alpha=0.6, label='Input')
                axs[i].plot(rec[local_idx].reshape(-1), color='green', alpha=0.8, label='Predicted')
                break
            offset += batch_size
        title = f'Panel {panel}' + (' (held out)' if panel == held_out_panel else '') + f', state={state_idx}'
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    fig.suptitle(f'Reconstruction — fold validated on {held_out_panel}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _normalization_mean_variance_axis(norm_stats, input_shape):
    '''Shapes a fold's feature_means/feature_stds for tf.keras.layers.Normalization
    against a model's (batch-excluded) input_shape.

    Two cases, depending on how many channels the raw feature array had when the stats
    were fit (see states_check.prepare_datastores):
    - mean/std already match input_shape exactly (e.g. (33, 2) for CNN_AE_features with
      include_benchmark=True and no differencing) -- every (feature, channel) pair got its
      own independently-fit stat, so every non-batch axis is a Normalization axis and the
      arrays are used as-is.
    - mean/std are a flat (n_feat,) array (fc_AE, or CNN_AE without benchmark / with
      diff_bench) -- a single per-feature stat that must broadcast across any extra
      channel axis, so it's reshaped onto just the feature axis.
    '''
    mean = norm_stats["feature_means"].astype(np.float32)
    variance = norm_stats["feature_stds"].astype(np.float32) ** 2

    if mean.shape == input_shape:
        axis = tuple(range(1, len(input_shape) + 1))
        return mean, variance, axis

    # Flat per-feature stat: broadcast onto the first non-batch axis (-1 for the flat
    # fc_AE shape (n_feat,), 1 for the CNN_AE shape (n_feat, n_channels)).
    axis = 1 if len(input_shape) > 1 else -1
    bshape = [1] * len(input_shape)
    target_idx = axis - 1 if axis > 0 else axis
    bshape[target_idx] = mean.shape[0]
    return mean.reshape(bshape), variance.reshape(bshape), axis


def build_ensemble_ae(fold_entries):
    '''Combines the per-fold cross-validation models into one averaging ensemble.

    Each fold's model was trained on features normalized with that fold's own mean/std
    (fit only on that fold's 3 training panels), so the ensemble can't just average raw
    model outputs -- it re-normalizes the shared raw input separately per branch, and
    de-normalizes each branch's reconstruction back to raw units (the reconstruction
    target is the normalized input itself, see states_check.normalize) before averaging.

    fold_entries: list of (panel_held_out, keras_model, norm_stats) tuples.
    '''
    input_shape = fold_entries[0][1].input_shape[1:]
    inp = tf.keras.Input(shape=input_shape, name="raw_input")

    shi_branches, rec_branches, latent_branches = [], [], []
    for panel, model, norm_stats in fold_entries:
        # "latent_space" is an internal KSparse layer, not one of model's declared
        # outputs (only sHI/reconstruction are). Fold it into a single 3-output model
        # up front, rather than calling `model` plus a second sub-model tapping the same
        # encoder layers -- two separate Model containers both tracking the same shared
        # layers breaks Keras's .keras save/reload (the shared layers' weights silently
        # drop on reload, since they end up double-tracked in the object graph).
        shi_idx = model.output_names.index("sHI")
        rec_idx = model.output_names.index("reconstruction")
        latent_output = model.get_layer("latent_space").output
        model = tf.keras.Model(inputs=model.input, outputs=[model.output[shi_idx], model.output[rec_idx], latent_output], name=model.name)
        shi_idx, rec_idx, latent_idx = 0, 1, 2

        # Every fold's model was built by the same factory function (build_CNN_AE_features
        # / build_fc_AE_features), so their internal layers (bench_comp, enc_dense,
        # latent_space, sHI, ...) all share identical names across folds. Keras's .keras
        # save format indexes weights by layer name across the *whole* nested tree, so
        # without uniquifying them here, sibling folds' same-named layers collide on
        # save and silently drop weights on reload. Renaming model.name alone (as
        # before) isn't enough -- every layer inside it needs a unique name too.
        prefix = f"fold_{panel}_"
        for layer in model.layers:
            layer.name = prefix + layer.name
        model.name = prefix + model.name

        mean, variance, norm_axis = _normalization_mean_variance_axis(norm_stats, input_shape)

        # Normalization stores mean/variance as weights (unlike a Lambda closing over
        # tf.constant, which Keras can't JSON-serialize), so the ensemble saves cleanly.
        norm_layer = tf.keras.layers.Normalization(axis=norm_axis, mean=mean, variance=variance, name=f"norm_{panel}")
        denorm_layer = tf.keras.layers.Normalization(axis=norm_axis, mean=mean, variance=variance, invert=True, name=f"denorm_{panel}")

        x_norm = norm_layer(inp)
        x_norm = ClipLayer(-10.0, 10.0, name=f"clip_{panel}")(x_norm)

        outs = model(x_norm)
        rec_denorm = denorm_layer(outs[rec_idx])

        shi_branches.append(outs[shi_idx])
        rec_branches.append(rec_denorm)
        latent_branches.append(outs[latent_idx])

    shi_avg = tf.keras.layers.Average(name="sHI")(shi_branches)
    rec_avg = tf.keras.layers.Average(name="reconstruction")(rec_branches)
    latent_avg = tf.keras.layers.Average(name="latent_space")(latent_branches)
    return tf.keras.Model(inputs=inp, outputs=[shi_avg, rec_avg, latent_avg], name="ensemble_ae")


def plot_ensemble_sHI(ensemble_model, ref_ds_dict, ref_norm_stats, States_dict, panels, save_path):
    '''sHI vs state predicted by the ensemble, one subplot per base panel.

    ref_ds_dict/ref_norm_stats: any single fold's ds_dict plus the norm_stats that produced
    it -- used only to recover each panel's raw (unnormalized) features, since the ensemble
    re-normalizes per branch internally and expects raw input.
    '''
    input_shape = ensemble_model.input_shape[1:]
    mean_b, variance_b, _ = _normalization_mean_variance_axis(ref_norm_stats, input_shape)
    std_b = np.sqrt(variance_b)

    shi_idx = ensemble_model.output_names.index("sHI")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, panel in enumerate(panels):
        states = States_dict[panel]
        all_sHI = []
        for x_norm, _ in ref_ds_dict[panel]:
            # The datastore yields features with shape (batch, n_feat) even for CNN_AE_features
            # (whose Input is (n_feat, n_channels)) -- Keras' fit/predict silently tolerates that
            # trailing-1-dim mismatch, but plain numpy broadcasting below does not.
            x_norm = np.asarray(x_norm).reshape((-1,) + input_shape)
            x_raw = x_norm * std_b + mean_b
            pred = ensemble_model.predict(x_raw, verbose=0)
            all_sHI.append(np.asarray(pred[shi_idx]).reshape(-1))
        sHI = np.concatenate(all_sHI)
        axs[i].scatter(states, sHI, alpha=0.7, s=15, color='purple')
        axs[i].set_title(f'Panel {panel}')
        axs[i].set_xlabel('State idx')
        axs[i].set_ylabel('sHI (ensemble)')
        axs[i].grid(True)
    fig.suptitle('Ensemble sHI vs State')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
