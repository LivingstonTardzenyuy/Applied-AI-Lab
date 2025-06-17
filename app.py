from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image  # Replaces keras.preprocessing.image
import numpy as np
import io  # For in-memory file handling
import os

# Azure-compatible model loading
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_2_augment_shuffles.h5')
model = load_model(model_path)

app = Flask(__name__)

class_labels = ['Normal', 'Pneumonia']

@app.route('/')
def home():
    return "Pneumonia Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Process image directly from memory
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB').resize((128, 128))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        pred_class = class_labels[int(prediction[0][0] > 0.5)]

        return jsonify({'prediction': pred_class})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
