# Original Architecture Credit: DeepFake-Video-Detection repository (InceptionV3 Feature Extractor + Sequential GRU Classifier)

import sys
import os
import json
import ssl

# Disable torch dynamo & optree C-extension attempts on Windows Python 3.14
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "-all"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cv2
import keras
from keras import layers

ssl._create_default_https_context = ssl._create_unverified_context

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def build_model(model_weights_path):
    inputs = layers.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name="input_9")
    mask = layers.Input(shape=(MAX_SEQ_LENGTH,), dtype="bool", name="input_10")

    x = layers.GRU(64, return_sequences=True, reset_after=True, name="gru_5")(inputs, mask=mask)
    x = layers.BatchNormalization(name="batch_normalization_190")(x)
    x = layers.GRU(32, return_sequences=True, reset_after=True, name="gru_6")(x)
    x = layers.BatchNormalization(name="batch_normalization_191")(x)
    x = layers.GRU(16, return_sequences=False, reset_after=True, name="gru_7")(x)
    x = layers.Dropout(0.5, name="dropout_3")(x, training=False)
    x = layers.Dense(32, activation="relu", name="dense_4")(x)
    x = layers.Dropout(0.5, name="dropout_4")(x, training=False)
    outputs = layers.Dense(1, activation="sigmoid", name="dense_5")(x)

    model = keras.Model(inputs=[inputs, mask], outputs=outputs)
    model.load_weights(model_weights_path)
    return model

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

        # Fallback to sequential read if seeking resulted in 0 frames
        if len(frames) == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                if max_frames > 0 and len(frames) == max_frames:
                    break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames, feature_extractor):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        if length > 0:
            batch_feats = feature_extractor.predict(batch[:length].astype("float32"), batch_size=length, verbose=0)
            frame_features[i, :length, :] = batch_feats
            frame_mask[i, :length] = 1

    return frame_features, frame_mask

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No video file path provided"}))
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(json.dumps({"error": f"File not found: {video_path}"}))
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "deepfake_video_model.h5")

    if not os.path.isfile(weights_path):
        print(json.dumps({"error": f"Model weights not found: {weights_path}"}))
        sys.exit(1)

    try:
        feature_extractor = build_feature_extractor()
        model = build_model(weights_path)

        frames = load_video(video_path)
        if len(frames) == 0:
            print(json.dumps({"error": "Could not read frames from video"}))
            sys.exit(1)

        frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
        T = int(np.sum(frame_mask[0]))
        if T == 0:
            print(json.dumps({"error": "No valid frames prepared"}))
            sys.exit(1)

        # Batch evaluation of temporal prefix sequences to compute true per-frame probabilities
        batch_features = np.zeros((T, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        batch_mask = np.zeros((T, MAX_SEQ_LENGTH), dtype="bool")
        for k in range(T):
            batch_features[k, :k+1, :] = frame_features[0, :k+1, :]
            batch_mask[k, :k+1] = True

        preds = model.predict([batch_features, batch_mask], verbose=0)
        frame_scores = [round(float(p[0]), 4) for p in preds]

        # The overall video prediction is the final sequence prediction
        prediction = float(preds[-1][0])
        fake_prob = prediction
        real_prob = 1.0 - fake_prob

        # Calculate raw logit
        eps = 1e-7
        clipped_p = np.clip(prediction, eps, 1.0 - eps)
        raw_logit = float(np.log(clipped_p / (1.0 - clipped_p)))

        # Binary decision with standard 0.50 threshold
        threshold = 0.50
        verdict = "FAKE" if prediction >= threshold else "REAL"
        video_verdict = "MANIPULATED" if verdict == "FAKE" else "AUTHENTIC"
        confidence = fake_prob if verdict == "FAKE" else real_prob

        out = {
            "verdict": verdict,
            "video_verdict": video_verdict,
            "prediction": f"LIKELY {video_verdict}",
            "confidence": round(confidence, 4),
            "avg_confidence": round(confidence, 4),
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "raw_logit": round(raw_logit, 4),
            "threshold": threshold,
            "frames_analyzed": T,
            "sequence_shape": [1, int(MAX_SEQ_LENGTH), int(NUM_FEATURES)],
            "frame_scores": frame_scores,
            "model": "CNN-GRU Temporal Model",
            "checkpoint": weights_path,
            "source": "CNN-GRU trained model (external, isolated env)",
            "fallback": False
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
