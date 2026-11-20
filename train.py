"""Train & compare CNN, EfficientNetB0, ResNet50, MobileNetV2, VGG16, VGG19, Xception on the merged dataset.
Grouped 5-fold CV; synthetic images used for TRAINING only; scores computed on ORIGINAL images only.
Run in Colab (GPU):  !python train.py"""
import os, sys, numpy as np, pandas as pd, tensorflow as tf
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras import layers, models, applications as A

DATA, IMG, BATCH, FOLDS = "merged_dataset", 224, 16, 5
EP_HEAD, EP_FT, USE_SYNTH = 25, 15, True
T, LOG = ["ph", "turbidity", "cod", "bod"], ["turbidity", "cod", "bod"]
MODELS = sys.argv[1:] or ["CNN", "EfficientNetB0", "ResNet50", "MobileNetV2", "VGG16", "VGG19", "Xception"]
tf.keras.utils.set_random_seed(42)

df = pd.read_csv(f"{DATA}/master.csv")
X = np.stack([np.asarray(Image.open(f"{DATA}/{f}").convert("RGB").resize((IMG, IMG))) for f in df.file]).astype("float32")
Y = df[T].copy(); Y[LOG] = np.log1p(Y[LOG]); Y = Y.values            # log-transform the heavy-tailed targets
orig = (df.is_synthetic == 0).values

def masked_mse(y, p):                                                # missing labels (NaN) are ignored, not dropped
    v, m = y[:, :4], y[:, 4:]
    return tf.reduce_sum(m * tf.square(v - p)) / tf.maximum(tf.reduce_sum(m), 1.0)

BASES = {"EfficientNetB0": (A.EfficientNetB0, lambda x: x), "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
         "MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input), "VGG16": (A.VGG16, A.vgg16.preprocess_input),
         "VGG19": (A.VGG19, A.vgg19.preprocess_input), "Xception": (A.Xception, A.xception.preprocess_input)}

def build(name):
    i = layers.Input((IMG, IMG, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(i); x = layers.RandomZoom(0.1)(x)   # no colour jitter: colour carries the signal
    if name == "CNN":
        x = layers.Rescaling(1 / 255)(x)
        for f in (32, 64, 128, 128):
            x = layers.MaxPooling2D()(layers.BatchNormalization()(layers.Conv2D(f, 3, padding="same", activation="relu")(x)))
        x = layers.GlobalAveragePooling2D()(x); base = None
    else:
        cls, pre = BASES[name]; base = cls(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3), pooling="avg")
        base.trainable = False; x = base(layers.Lambda(pre)(x), training=False)
    x = layers.Dense(64, activation="relu")(layers.Dropout(0.3)(x))
    return models.Model(i, layers.Dense(4)(x)), base

def fit(model, base, xt, yt, lr_head):
    model.compile(tf.keras.optimizers.Adam(lr_head if base else 1e-3), loss=masked_mse)
    model.fit(xt, yt, epochs=EP_HEAD if base else EP_HEAD + EP_FT, batch_size=BATCH, verbose=0)
    if base:                                                          # fine-tune the last 30 layers (BatchNorm stays frozen)
        base.trainable = True
        for l in base.layers[:-30]: l.trainable = False
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization): l.trainable = False
        model.compile(tf.keras.optimizers.Adam(1e-5), loss=masked_mse)
        model.fit(xt, yt, epochs=EP_FT, batch_size=BATCH, verbose=0)

results, groups = [], df.group_id.values
for name in MODELS:
    oof = np.full(Y.shape, np.nan); base_pred = np.full(Y.shape, np.nan)
    for k, (tr, va) in enumerate(GroupKFold(FOLDS).split(X, groups=groups)):
        tr = tr if USE_SYNTH else tr[orig[tr]]; va = va[orig[va]]
        mu, sd = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0)             # statistics from TRAIN rows only
        Z = (Y[tr] - mu) / sd; yt = np.hstack([np.nan_to_num(Z), (~np.isnan(Z)).astype("float32")])
        tf.keras.backend.clear_session(); model, base = build(name); fit(model, base, X[tr], yt, 1e-3)
        oof[va] = model.predict(X[va], verbose=0) * sd + mu; base_pred[va] = mu   # base_pred = "always predict the mean"
        print(name, "fold", k + 1, "done", flush=True)
    for j, t in enumerate(T):
        ok = orig & ~np.isnan(Y[:, j])
        results.append(dict(model=name, target=t + (" (log)" if t in LOG else ""), R2=r2_score(Y[ok, j], oof[ok, j]),
                            R2_mean_baseline=r2_score(Y[ok, j], base_pred[ok, j]), MAE=mean_absolute_error(Y[ok, j], oof[ok, j])))
    pd.DataFrame(results).to_csv("results.csv", index=False)
r = pd.DataFrame(results); print(r.pivot(index="model", columns="target", values="R2").round(3).to_string())
