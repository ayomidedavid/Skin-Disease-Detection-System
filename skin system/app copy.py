from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('hybrid_model.h5')

# Updated preprocessing function
def preprocess_image(image):
    # Resize the image to the target size (28x28)
    image = image.resize((28, 28))
    
    # Convert the image to a numpy array
    image = np.array(image)
    
    # Normalize the image
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    file = request.files['image'].read()
    
    # Convert the image to a PIL Image
    image = Image.open(io.BytesIO(file))
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction
    prediction = model.predict(preprocessed_image)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Render the result page with the predicted class
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
