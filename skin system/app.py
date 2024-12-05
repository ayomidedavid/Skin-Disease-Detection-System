from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = tf.keras.models.load_model('hybrid_model.h5')

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    image = Image.open(file)

    # Ensure the upload directory exists
    upload_dir = os.path.join(app.static_folder, 'uploads')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the image for displaying later
    image_path = os.path.join(upload_dir, filename)
    image.save(image_path)

    # Debugging print statements
    print(f"Image saved at: {image_path}")

    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    classes = {
        0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
        1: ('bcc', 'basal cell carcinoma'),
        2: ('bkl', 'benign keratosis-like lesions'),
        3: ('df', 'dermatofibroma'),
        4: ('nv', 'melanocytic nevi'),
        5: ('vasc', 'pyogenic granulomas and hemorrhage'),
        6: ('mel', 'melanoma')
    }

    predicted_class_name = classes.get(predicted_class, ('unknown', 'Unknown lesion type'))[1]

    # Pass the relative path for the image
    image_relative_path = os.path.join('uploads', filename)

    # Debugging print statements
    print(f"Predicted class: {predicted_class_name}")
    print(f"Image relative path: {image_relative_path}")

    # Replace backslashes with forward slashes
    image_relative_path = os.path.join('uploads', file.filename).replace('\\', '/')

    # Render the template with the corrected image path
    return render_template('result.html', prediction=predicted_class_name, image_path=image_relative_path)


if __name__ == '__main__':
    app.run(debug=True)
