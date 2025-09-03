# Predicción de Neumonía — Compatible con Keras 3 (Colab) + TensorFlow 2.17

## Requisitos
- Python 3.10 o 3.11
- Entorno virtual (`venv`).

## Pasos (Windows PowerShell)
```powershell
cd "C:\ruta\proyecto-neumonia-keras3"
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Copia tu modelo (de Colab) como best_effnetv2.keras en la raíz
python app.py
```
Abrir: http://localhost:4000  
Salud: http://localhost:4000/health

## Estructura
```
proyecto-neumonia-keras3/
├─ app.py
├─ requirements.txt
├─ best_effnetv2.keras   # <-- coloca aquí tu modelo
├─ templates/
│  ├─ index.html
│  └─ records.html
├─ static/
│  └─ style.css
└─ uploads/
```
