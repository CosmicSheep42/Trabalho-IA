from flask import Flask, request, render_template, send_from_directory
import os
import joblib
import numpy as np

from tools.feature_extractor import extract_feature_vgg16

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(APP_ROOT, 'resultados', 'simpsons_ensemble_model.pkl')
SCALER_PATH = os.path.join(APP_ROOT, 'resultados', 'simpsons_scaler.pkl')
CLASSES_PATH = os.path.join(APP_ROOT, 'resultados', 'classes_names.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
scaler = None
classes = None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = joblib.load(CLASSES_PATH)
except Exception as e:
    print(f"AVISO: Não foi possível carregar model/scaler/classes: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or classes is None:
        return render_template('result.html', error='Modelo ou artefatos não encontrados. Execute o treinamento primeiro.')

    if 'image' not in request.files:
        return render_template('result.html', error='Nenhum arquivo enviado.')

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('result.html', error='Arquivo inválido. Use png/jpg/jpeg.')

    filename = file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    feats = extract_feature_vgg16([save_path])
    if feats is None or len(feats) == 0:
        return render_template('result.html', error='Falha ao extrair features da imagem.')

    try:
        feats_scaled = scaler.transform(feats)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(feats_scaled)
            top_idx = np.argmax(probs, axis=1)[0]
            confidence = float(probs[0, top_idx]) * 100.0
            pred = model.classes_[top_idx] if hasattr(model, 'classes_') else None
        else:
            pred = model.predict(feats_scaled)[0]
            confidence = None

        pred_name = None
        if isinstance(pred, (int, np.integer)):
            pred_name = classes[pred]
        else:
            pred_name = str(pred)

        return render_template('result.html', filename=filename, prediction=pred_name, confidence=confidence)
    except Exception as e:
        return render_template('result.html', error=f'Erro durante previsão: {e}')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
