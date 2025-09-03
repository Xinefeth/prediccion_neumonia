🫁 Sistema de Predicción de Neumonía

Este proyecto implementa un sistema de detección de neumonía a partir de radiografías de tórax usando Deep Learning.
El modelo está basado en EfficientNetV2 y fue entrenado en Google Colab.
Además, cuenta con una interfaz web construida con Flask, que permite subir imágenes, obtener predicciones y registrar los resultados en una base de datos SQLite.

📌 Contenido del repositorio
proyecto-neumonia/
│── app.py                # Aplicación Flask principal
│── best_effnetv2.keras   # Modelo entrenado de TensorFlow/Keras
│── requirements.txt      # Dependencias del proyecto
│── templates/            # Archivos HTML (frontend)
│   ├── index.html        # Interfaz principal (subir RX y predecir)
│   └── records.html      # Listado de registros con predicciones
│── static/               # Archivos estáticos (CSS, JS)
│── uploads/              # Carpeta donde se guardan las imágenes subidas
│── neumonia.db           # Base de datos SQLite (se crea automáticamente)
└── README.md             # Documentación del proyecto

⚙️ Tecnologías utilizadas

Python 3.10+

TensorFlow 2.17.0 / Keras 3.4.1

Flask 3.0.3

SQLAlchemy 2.0

Pillow (procesamiento de imágenes)

Bootstrap 5 (frontend responsivo)

🧠 Modelo de Deep Learning

El modelo fue entrenado en Google Colab usando EfficientNetV2B0, con las siguientes características:

Dataset: Radiografías de tórax (Normal / Neumonía).

Preprocesamiento:

Redimensionamiento a 224x224.

Normalización con efficientnet_v2.preprocess_input.

Fine-tuning sobre capas superiores de EfficientNetV2.

Métrica principal: Accuracy y Confusion Matrix.

En Colab, el modelo se probó y alcanzó precisiones superiores al 97% en imágenes de prueba:

🚀 Ejecución en local
1️⃣ Clonar el repositorio
git clone https://github.com/Xinefeth/prediccion_neumonia.git
cd prediccion_neumonia

2️⃣ Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate

3️⃣ Instalar dependencias
pip install -r requirements.txt

4️⃣ Colocar el modelo

Copia tu archivo entrenado best_effnetv2.keras dentro de la carpeta raíz del proyecto.

5️⃣ Ejecutar la app
python app.py


La app correrá en:
👉 http://localhost:4000

🖥️ Funcionalidades

✅ Subir radiografía de tórax.
✅ Obtener predicción automática (Normal / Neumonía).
✅ Guardar el resultado en la base de datos.
✅ Consultar registros anteriores (nombre del paciente, edad, resultado, probabilidad, imagen).

📊 Ejemplo de predicción

Entrada: Radiografía de tórax.

Salida:

{
  "id": 1,
  "label": "PNEUMONIA",
  "prob": 0.9873,
  "image_url": "/uploads/imagen.png"
}

📌 Notas

El archivo uploads/ y la base neumonia.db se generan automáticamente al usar la app.

No se debe versionar .venv/, uploads/ ni neumonia.db. Usa el archivo .gitignore para excluirlos.

El modelo puede ser reemplazado por cualquier otro compatible con TensorFlow/Keras.

👨‍💻 Autor

Proyecto desarrollado por Diego Francesco Jara Tirado
📧 jaratiradodiego@gmail.com