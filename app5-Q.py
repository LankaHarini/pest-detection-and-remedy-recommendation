# ================= IMPORTS =================
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib
import json
import pennylane as qml
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ================= FIREBASE INITIALIZATION =================
if not firebase_admin._apps:
    cred = credentials.Certificate(r"D:\WEB\Quantum Saved Extensions\APK File.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ================= PATH =================
MODEL_DIR = r"D:\WEB\Quantum Saved Extensions"

# ================= LOAD MODELS =================
binary_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "binary_model.h5"),
    compile=False
)

multi_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "multi_model.h5"),
    compile=False
)

scaler_binary = joblib.load(os.path.join(MODEL_DIR, "scaler_binary.pkl"))
scaler_multi = joblib.load(os.path.join(MODEL_DIR, "scaler_multi.pkl"))
pca_binary = joblib.load(os.path.join(MODEL_DIR, "pca_binary.pkl"))
quantum_params_binary = np.load(os.path.join(MODEL_DIR, "quantum_params_binary.npy"))

with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    class_names = json.load(f)

# ================= LOAD NASNET =================
IMG_SIZE = 224

base_model = tf.keras.applications.NASNetMobile(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

feature_extractor = tf.keras.Model(
    base_model.input,
    tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
)

# ================= QUANTUM FUNCTION =================
def quantum_feature_map_binary(X, params):
    n_samples = X.shape[0]
    n_qubits = X.shape[1]
    quantum_features = np.zeros((n_samples, n_qubits * 2))

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(x, p, measurement_qubit):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        for layer in range(2):
            for i in range(n_qubits):
                qml.RY(p[layer, i, 0], wires=i)
                qml.RZ(p[layer, i, 1], wires=i)

            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        return (
            qml.expval(qml.PauliZ(measurement_qubit)),
            qml.expval(qml.PauliX(measurement_qubit))
        )

    for i in range(n_samples):
        features = []
        for qubit in range(n_qubits):
            z_exp, x_exp = quantum_circuit(X[i], params, qubit)
            features.extend([z_exp, x_exp])
        quantum_features[i] = features

    return quantum_features

# ================= PREPROCESS =================
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Quantum Maize Disease Detector", layout="centered")

st.title("🌽 Maize Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

# ================= MAIN LOGIC =================
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_processed = preprocess_image(img)
    st.image(img_processed, caption="Uploaded Image", use_column_width=True)

    # ========== FEATURE EXTRACTION ==========
    feat = feature_extractor.predict(np.expand_dims(img_processed, 0), verbose=0)

    # ===== Binary Check =====
    feat_scaled_bin = scaler_binary.transform(feat)
    feat_pca = pca_binary.transform(feat_scaled_bin)
    quantum_feat = quantum_feature_map_binary(feat_pca, quantum_params_binary)
    prob_binary = binary_model.predict(quantum_feat, verbose=0)[0][0]

    # ===== Multi-class Prediction =====
    feat_scaled_multi = scaler_multi.transform(feat)
    probs = multi_model.predict(feat_scaled_multi, verbose=0)[0]
    pred_class = np.argmax(probs)

    final_label = class_names[pred_class]
    confidence = float(probs[pred_class])

    # ================= FETCH REMEDY FROM FIREBASE =================
    remedy_doc = db.collection("remedies").document(final_label).get()

    if remedy_doc.exists:
        remedy_data = remedy_doc.to_dict()
    else:
        remedy_data = None

    # ================= STORE PREDICTION =================
    db.collection("predictions").add({
        "disease": final_label,
        "confidence": confidence,
        "timestamp": datetime.now()
    })

    # ================= FINAL OUTPUT =================
    st.markdown("### 🔍 Prediction Result")

    if prob_binary < 0.5:
        st.success("🌿 Healthy Leaf")
    else:
        st.error(f"🦠 {final_label}")
        st.write(f"Confidence: {confidence:.3f}")

        if remedy_data:
            st.markdown("## 💊 Recommended Treatment")
            st.info(f"""
            **Remedy:** {remedy_data['remedy']}  
            **Dosage:** {remedy_data['dosage']}  
            **Prevention:** {remedy_data['prevention']}
            """)

