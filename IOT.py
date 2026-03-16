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
from firebase_admin import credentials, firestore, db
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Quantum Maize Disease Detector", layout="centered")
st_autorefresh(interval=5000, key="sensor_refresh")

MODEL_DIR = r"D:\WEB\Quantum Saved Extensions"
IMG_SIZE = 224

# ================= FIREBASE =================
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(r"D:\WEB\Quantum Saved Extensions\APK File.json")
        firebase_admin.initialize_app(cred, {
            "databaseURL": "https://maize-711be-default-rtdb.firebaseio.com/"
        })
    return firestore.client()

firestore_db = init_firebase()

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    binary_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "binary_model.h5"), compile=False
    )
    multi_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "multi_model.h5"), compile=False
    )
    scaler_binary = joblib.load(os.path.join(MODEL_DIR, "scaler_binary.pkl"))
    scaler_multi  = joblib.load(os.path.join(MODEL_DIR, "scaler_multi.pkl"))
    pca_binary    = joblib.load(os.path.join(MODEL_DIR, "pca_binary.pkl"))
    quantum_params_binary = np.load(os.path.join(MODEL_DIR, "quantum_params_binary.npy"))
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    return (binary_model, multi_model, scaler_binary,
            scaler_multi, pca_binary, quantum_params_binary, class_names)

(binary_model, multi_model, scaler_binary,
 scaler_multi, pca_binary, quantum_params_binary,
 class_names) = load_models()

# ================= LOAD NASNET =================
@st.cache_resource
def load_feature_extractor():
    base_model = tf.keras.applications.NASNetMobile(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    return tf.keras.Model(
        base_model.input,
        tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    )

feature_extractor = load_feature_extractor()

# ================= QUANTUM FUNCTION =================
def quantum_feature_map_binary(X, params):
    n_samples = X.shape[0]
    n_qubits  = X.shape[1]
    quantum_features = np.zeros((n_samples, n_qubits * 2))
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def quantum_circuit(x, p, m):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for layer in range(2):
            for i in range(n_qubits):
                qml.RY(p[layer, i, 0], wires=i)
                qml.RZ(p[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(qml.PauliZ(m)), qml.expval(qml.PauliX(m))

    for i in range(n_samples):
        features = []
        for q in range(n_qubits):
            z, x = quantum_circuit(X[i], params, q)
            features.extend([z, x])
        quantum_features[i] = features

    return quantum_features

# ================= TITLE =================
st.title("🌽 A Hybrid IoT–Cloud Architecture with Quantum Computing Support for Early Detection of Pests and Diseases in Maize Crops")

# ================= IOT SENSOR DATA =================
st.markdown("## 📡 Live IoT Sensor Data")

# ---- Always fetch fresh from Firebase (no caching) ----
sensor_data      = {}
last_updated     = "N/A"
device_connected = False
fetch_error      = None

try:
    ref    = db.reference("sensor_data/current")
    fetched = ref.get()
    if fetched:
        sensor_data  = fetched
        raw_ts       = sensor_data.get("LastUpdated", "N/A")
        last_updated = str(raw_ts).strip().strip('"').strip("'")

        # Consider connected if updated within last 300 seconds (5 minutes)
        if last_updated != "N/A":
            try:
                last_dt     = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                seconds_ago = (datetime.now() - last_dt).total_seconds()
                device_connected = seconds_ago <= 15
            except Exception:
                device_connected = True  # If parse fails, assume connected since data exists
    else:
        fetch_error = "No data returned from Firebase"
except Exception as e:
    fetch_error = str(e)

# ---- Connection status ----
if fetch_error:
    st.error(f"🔴 Device Not Connected — {fetch_error}")
elif device_connected:
    st.success("🟢 Device Connected")
else:
    st.error("🔴 Device Not Connected (No recent updates)")

# ---- Sensor values (always shown) ----
temp     = sensor_data.get("Temperature",  "N/A")
humidity = sensor_data.get("Humidity",     "N/A")
soil     = sensor_data.get("SoilMoisture", "N/A")
n_val    = sensor_data.get("N", "N/A")
p_val    = sensor_data.get("P", "N/A")
k_val    = sensor_data.get("K", "N/A")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("🌡️ Temperature",   f"{temp} °C"    if temp     != "N/A" else "N/A")
col2.metric("💧 Humidity",       f"{humidity} %" if humidity != "N/A" else "N/A")
col3.metric("🌱 Soil Moisture",  f"{soil}"       if soil     != "N/A" else "N/A")
col4.metric("🧪 Nitrogen (N)",   f"{n_val}"      if n_val    != "N/A" else "N/A")
col5.metric("🧪 Phosphorus (P)", f"{p_val}"      if p_val    != "N/A" else "N/A")
col6.metric("🧪 Potassium (K)",  f"{k_val}"      if k_val    != "N/A" else "N/A")

# ---- Last Updated (always shown at bottom) ----
st.caption(f"🕒 Last Updated: {last_updated}")

st.markdown("---")

# ================= DISEASE PREDICTION =================
st.markdown("## 🌿 Upload Leaf Image")
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0

    st.image(img, caption="Uploaded Image", use_column_width=True)

    feat = feature_extractor.predict(np.expand_dims(img, 0), verbose=0)

    feat_scaled_bin = scaler_binary.transform(feat)
    feat_pca        = pca_binary.transform(feat_scaled_bin)
    quantum_feat    = quantum_feature_map_binary(feat_pca, quantum_params_binary)
    prob_binary     = binary_model.predict(quantum_feat, verbose=0)[0][0]

    feat_scaled_multi = scaler_multi.transform(feat)
    probs             = multi_model.predict(feat_scaled_multi, verbose=0)[0]
    pred_class        = np.argmax(probs)

    final_label = class_names[pred_class]
    confidence  = float(probs[pred_class])

    st.markdown("### 🔍 Prediction Result")

    if prob_binary < 0.5:
        st.success("🌿 Healthy Leaf")
    else:
        st.error(f"🦠 {final_label}")
        st.write(f"Confidence: {confidence:.3f}")

        remedy_doc = firestore_db.collection("remedies").document(final_label).get()
        if remedy_doc.exists:
            remedy_data = remedy_doc.to_dict()
            st.markdown("## 💊 Recommended Treatment")
            st.info(f"""
**Remedy:** {remedy_data.get('remedy', 'N/A')}  
**Dosage:** {remedy_data.get('dosage', 'N/A')}  
**Prevention:** {remedy_data.get('prevention', 'N/A')}
            """)

    firestore_db.collection("predictions").add({
        "disease":    final_label,
        "confidence": confidence,
        "timestamp":  datetime.now()
    })