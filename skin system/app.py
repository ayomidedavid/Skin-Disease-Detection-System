from flask import Flask, request, render_template, url_for, redirect, session, flash
from db import get_db_connection
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure random value

def create_tables():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                image VARCHAR(255) NOT NULL,
                prediction VARCHAR(255) NOT NULL,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )''')
        connection.commit()
    finally:
        connection.close()

# @app.before_first_request
# def before_first_request():
#     create_tables()
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT * FROM users WHERE username=%s', (username,))
                if cursor.fetchone():
                    flash('Username already exists!')
                    return render_template('signup.html')
                cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, password))
            connection.commit()
            flash('Signup successful! Please login.')
            return redirect('/login')
        finally:
            connection.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, password))
                user = cursor.fetchone()
                if user:
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    return redirect('/dashboard')
                else:
                    flash('Invalid credentials!')
        finally:
            connection.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

model = tf.keras.models.load_model('themodel.h5')

def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/')
def home():
    if 'user_id' in session:
        return redirect('/dashboard')
    return redirect('/login')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session.get('username'))


@app.route('/predict', methods=['POST'])
@login_required
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

    # Save prediction to history
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute('INSERT INTO history (user_id, image, prediction) VALUES (%s, %s, %s)',
                           (session['user_id'], filename, predicted_class_name))
        connection.commit()
    finally:
        connection.close()

    image_relative_path = os.path.join('uploads', filename).replace('\\', '/')
    return render_template('result.html', prediction=predicted_class_name, image_path=image_relative_path)

@app.route('/history')
@login_required
def history():
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT date, image, prediction FROM history WHERE user_id=%s ORDER BY date DESC', (session['user_id'],))
            records = cursor.fetchall()
    finally:
        connection.close()
    return render_template('history.html', username=session.get('username'), history=records)


if __name__ == '__main__':
    create_tables()
    app.run(debug=True)
