import os
import traceback
from datetime import datetime
from uuid import uuid4

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2
from flask import Flask, request, render_template, url_for, jsonify, send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from PIL import Image, ImageOps


# Menos ruido de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Config ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "best_effnetv2.keras")  # ajusta si tu archivo se llama distinto
IMG_SIZE = (224, 224)  # usa el tamaño que entrenaste
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # respeta el orden del entrenamiento

# --- App ---
app = Flask(__name__)
CORS(app)

# --- DB ---
class Base(DeclarativeBase):
    pass

class Record(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True)
    patient_name = Column(String(120))
    document_id = Column(String(60))
    age = Column(Integer)
    notes = Column(String(500))
    image_path = Column(String(255))
    pred_label = Column(String(32))
    prob = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(f"sqlite:///{os.path.join(BASE_DIR,'neumonia.db')}", echo=False, future=True)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# --- Carga compatible (Keras 3 / tf.keras) ---
# --- Carga de modelo (exacta a tu Colab) ---
model = None
def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
        print(f"🧠 Cargando modelo (tf.keras) desde: {MODEL_PATH}", flush=True)
        model = tf.keras.models.load_model(MODEL_PATH)  # <- igual que en tu notebook
        print(f"✅ Modelo cargado | TF={tf.__version__}", flush=True)
    return model



def _load_model_any(path: str):
    # 1) Keras 3: recomendado para modelos guardados en Colab moderno
    try:
        import keras
        try:
            return keras.saving.load_model(path)
        except Exception:
            return keras.models.load_model(path)
    except Exception:
        pass
    # 2) Fallback: tf.keras (modelos guardados con TF 2.x)
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
        print(f"🧠 Cargando modelo desde: {MODEL_PATH}", flush=True)
        model = _load_model_any(MODEL_PATH)
        try:
            import tensorflow as tf, keras
            print(f"✅ Modelo cargado | TF={tf.__version__} | Keras={keras.__version__}", flush=True)
        except Exception:
            pass
    return model

# --- Preprocesamiento (igual que entrenamiento) ---
def preprocess_image(pil_img, target_size=IMG_SIZE):
    # Corrige orientación por EXIF y fija modo de reescalado
    img = ImageOps.exif_transpose(pil_img).convert("RGB").resize(
        target_size, resample=Image.BICUBIC
    )
    arr = np.array(img).astype("float32")
    arr = efficientnet_v2.preprocess_input(arr)   # <- como en tu Colab
    return np.expand_dims(arr, axis=0)

def predict_image(pil_img):
    x = preprocess_image(pil_img, IMG_SIZE)
    probs = get_model().predict(x, verbose=0)[0]
    if probs.shape[0] == 1:            # por si el modelo fuera sigmoide binaria
        p = float(probs[0]); probs = np.array([1.0 - p, p])
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


@app.route("/api/debug_predict", methods=["POST"])
def debug_predict():
    try:
        f = request.files["file"]
        img = Image.open(f.stream)
        x = preprocess_image(img, IMG_SIZE)
        probs = get_model().predict(x, verbose=0)[0].tolist()
        return jsonify({"class_names": CLASS_NAMES, "probs": probs, "argmax": int(np.argmax(probs))})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# --- Vistas ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/records")
def records_view():
    with SessionLocal() as db:
        items = db.query(Record).order_by(Record.created_at.desc()).all()
    return render_template("records.html", items=items)

# --- API ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Falta el archivo 'file'."}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Nombre de archivo vacío."}), 400

        ext = os.path.splitext(f.filename)[1].lower() or ".png"
        filename = f"{uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        f.save(save_path)

        pil_img = Image.open(save_path)
        label, prob = predict_image(pil_img)

        with SessionLocal() as db:
            rec = Record(
                patient_name=request.form.get("patient_name", "").strip() or "Sin nombre",
                document_id=request.form.get("document_id", "").strip(),
                age=int(request.form.get("age", 0) or 0),
                notes=request.form.get("notes", "").strip(),
                image_path=f"/uploads/{filename}",
                pred_label=label,
                prob=prob,
            )
            db.add(rec)
            db.commit()
            rec_id = rec.id

        return jsonify({
            "id": rec_id,
            "label": label,
            "prob": round(prob, 4),
            "image_url": url_for("serve_upload", filename=filename)
        })
    except Exception as e:
        print("❌ Error en /api/predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/records", methods=["GET"])
def api_records():
    with SessionLocal() as db:
        items = db.query(Record).order_by(Record.created_at.desc()).all()
    return jsonify([{
        "id": r.id,
        "patient_name": r.patient_name,
        "document_id": r.document_id,
        "age": r.age,
        "notes": r.notes,
        "image_path": r.image_path,
        "pred_label": r.pred_label,
        "prob": r.prob,
        "created_at": r.created_at.isoformat()
    } for r in items])

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    })

if __name__ == "__main__":
    print("🚀 Iniciando servidor Flask en http://localhost:4000", flush=True)
    app.run(host="0.0.0.0", port=4000, debug=True, use_reloader=False)
