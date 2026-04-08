from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATASET_PATH = BASE_DIR / "dataset" / "emotions.csv"


st.set_page_config(
    page_title="EEG Emotion Classifier",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "best_model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    selector = joblib.load(ARTIFACT_DIR / "feature_selector.pkl")
    label_encoder = joblib.load(ARTIFACT_DIR / "label_encoder.pkl")
    selected_features = json.loads((ARTIFACT_DIR / "selected_features.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    return model, scaler, selector, label_encoder, selected_features, metrics


@st.cache_data
def load_reference_dataset():
    df = pd.read_csv(DATASET_PATH)
    df = df.rename(columns=lambda c: c.strip())
    if "# mean_0_a" in df.columns:
        df = df.rename(columns={"# mean_0_a": "mean_0_a"})
    return df


def prepare_features(
    df_input: pd.DataFrame,
    scaler,
    selector,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    df = df_input.copy()
    df = df.rename(columns=lambda c: c.strip())
    if "# mean_0_a" in df.columns:
        df = df.rename(columns={"# mean_0_a": "mean_0_a"})

    extra_cols = [c for c in df.columns if c not in selected_features and c != "label"]
    missing_cols = [c for c in selected_features if c not in df.columns]

    feature_frame = df.drop(columns=["label"], errors="ignore").copy()
    for col in missing_cols:
        feature_frame[col] = np.nan

    feature_frame = feature_frame.reindex(columns=selected_features)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")

    if feature_frame.isna().any().any():
        fill_values = getattr(scaler, "mean_", np.zeros(len(selected_features)))
        feature_frame = feature_frame.fillna(pd.Series(fill_values, index=selected_features))

    scaled = pd.DataFrame(
        scaler.transform(feature_frame),
        columns=selected_features,
        index=feature_frame.index,
    )

    if isinstance(selector, dict):
        transformed = scaled
    else:
        transformed_array = selector.transform(scaled)
        transformed_columns = (
            selected_features
            if getattr(selector, "get_support", None) is None
            else list(np.array(selected_features)[selector.get_support()])
        )
        transformed = pd.DataFrame(
            transformed_array,
            columns=transformed_columns,
            index=scaled.index,
        )

    return feature_frame, transformed, missing_cols, extra_cols


def predict_dataframe(df_input: pd.DataFrame):
    model, scaler, selector, label_encoder, selected_features, _ = load_artifacts()
    raw_features, transformed, missing_cols, extra_cols = prepare_features(
        df_input=df_input,
        scaler=scaler,
        selector=selector,
        selected_features=selected_features,
    )
    pred_idx = model.predict(transformed)
    pred_label = label_encoder.inverse_transform(pred_idx)
    pred_proba = model.predict_proba(transformed)
    proba_df = pd.DataFrame(pred_proba, columns=label_encoder.classes_, index=df_input.index)

    result = df_input.copy()
    result["predicted_label"] = pred_label
    result["prediction_confidence"] = proba_df.max(axis=1).round(6)
    for cls in label_encoder.classes_:
        result[f"proba_{cls.lower()}"] = proba_df[cls].round(6)

    return result, raw_features, transformed, missing_cols, extra_cols


model, scaler, selector, label_encoder, selected_features, metrics = load_artifacts()

st.title("EEG Emotion Classification")
st.caption("Streamlit app untuk inferensi menggunakan artifact model terbaik hasil training sebelumnya.")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Model", metrics["best_configuration"]["model"])
col_b.metric("Scenario", metrics["best_configuration"]["scenario"])
col_c.metric("Macro F1", f'{metrics["best_configuration"]["f1_macro"]:.4f}')
col_d.metric("CV F1 Mean", f'{metrics["best_configuration"]["cv_f1_mean"]:.4f}')

dataset_available = DATASET_PATH.exists()

with st.sidebar:
    st.header("Data Source")
    mode_options = ["Upload CSV"]
    if dataset_available:
        mode_options.insert(0, "Use bundled dataset")
    mode = st.radio("Pilih sumber data", options=mode_options, index=0)

    st.markdown("Artifact directory")
    st.code(str(ARTIFACT_DIR))

    st.markdown("Expected features")
    st.write(f"{len(selected_features)} fitur numerik")


if mode == "Use bundled dataset" and dataset_available:
    source_df = load_reference_dataset()
    with st.expander("Preview dataset lokal", expanded=False):
        st.dataframe(source_df.head(10), use_container_width=True)
else:
    uploaded = st.file_uploader("Upload file CSV untuk prediksi", type=["csv"])
    if uploaded is None:
        st.info("Upload CSV untuk mulai prediksi.")
        st.stop()
    source_df = pd.read_csv(uploaded)


st.subheader("Inference")

if source_df.empty:
    st.warning("Data kosong.")
    st.stop()

results_df, raw_feature_df, transformed_df, missing_cols, extra_cols = predict_dataframe(source_df)

warn_msgs = []
if missing_cols:
    warn_msgs.append(f"{len(missing_cols)} kolom fitur tidak ditemukan dan diisi fallback dari statistik scaler.")
if extra_cols:
    warn_msgs.append(f"{len(extra_cols)} kolom ekstra diabaikan.")
if warn_msgs:
    for msg in warn_msgs:
        st.warning(msg)

left, right = st.columns([1.2, 1])

with left:
    st.markdown("**Prediction Results**")
    st.dataframe(results_df.head(50), use_container_width=True)

with right:
    st.markdown("**Predicted Class Distribution**")
    pred_counts = results_df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
    st.bar_chart(pred_counts.set_index("label"))


if "label" in source_df.columns:
    st.subheader("Evaluation on Provided Labels")
    y_true = source_df["label"].astype(str).to_numpy()
    y_pred = results_df["predicted_label"].astype(str).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    st.metric("Accuracy on current input", f"{acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(label_encoder.classes_))
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    st.dataframe(cm_df, use_container_width=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


st.subheader("Single Sample Inspection")
row_idx = st.number_input(
    "Pilih index baris",
    min_value=0,
    max_value=max(len(results_df) - 1, 0),
    value=0,
    step=1,
)

selected_row = results_df.iloc[int(row_idx)]
prob_cols = [f"proba_{cls.lower()}" for cls in label_encoder.classes_]
prob_display = pd.DataFrame(
    {
        "class": list(label_encoder.classes_),
        "probability": [selected_row[col] for col in prob_cols],
    }
).sort_values("probability", ascending=False)

col_1, col_2 = st.columns([1, 1])
with col_1:
    st.write(selected_row[["predicted_label", "prediction_confidence"]])
    st.dataframe(prob_display, use_container_width=True)
with col_2:
    st.dataframe(raw_feature_df.iloc[[int(row_idx)]].T.head(40), use_container_width=True)


csv_bytes = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download prediction results as CSV",
    data=csv_bytes,
    file_name="eeg_emotion_predictions.csv",
    mime="text/csv",
)

with st.expander("Model metadata", expanded=False):
    st.json(metrics)
