from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = load_model('model/model_2_augment_shuffles.h5')

app = Flask(__name__)

# Define class labels
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

    # Save the file
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))  # adjust to your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_labels[int(prediction[0][0] > 0.5)]  # binary classification

    return jsonify({'prediction': pred_class})

if __name__ == '__main__':
    app.run(debug=True)
