import os
import numpy as np
import cv2
import base64
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
from keras.utils import custom_object_scope
import keras

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = "static/uploads"
CAMERA_FOLDER = "static/camera"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["CAMERA_FOLDER"] = CAMERA_FOLDER

# Ensure directories exist
for folder in [UPLOAD_FOLDER, CAMERA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load Blood Group Model
BLOOD_MODEL_PATH = "models/blood_group_classifier_final.h5"
custom_objects = {"DTypePolicy": tf.keras.mixed_precision.Policy}

with custom_object_scope(custom_objects):
    try:
        blood_model = keras.models.load_model(BLOOD_MODEL_PATH, compile=False)
        print("✅ Blood Group Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading Blood Group model: {e}")
        blood_model = None

# Load Gender Classification Model
GENDER_MODEL_PATH = "models/GenderClassifyMF.h5"

try:
    gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH, compile=False)
    print("✅ Gender Classification Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Gender model: {e}")
    gender_model = None

# Load Eye Disease Model
EYE_MODEL_PATH = "models/model_cnn_fyp.keras"

try:
    eye_model = keras.models.load_model(EYE_MODEL_PATH, compile=False)
    print("✅ Eye Disease Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading Eye model: {e}")
    eye_model = None

# Classes
BLOOD_CLASSES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
EYE_CLASSES = [
    'Central Serous Chorioretinopathy-Color Fundus', 
     'Healthy',
    'Diabetic Retinopathy', 
    'Disc Edema', 
    'Glaucoma', 
    'Retinitis Pigmentosa', 
    'Retinal Detachment', 
    'Pterygium', 
    'Myopia', 
    'Macular Scar'
]

# Allowed file
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess for Blood Model
def preprocess_blood_image(image):
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Preprocess for Gender Model
def preprocess_gender_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((96, 96))
    img = np.array(img) / 255.0

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    img = np.expand_dims(img, axis=0)

    if gender_model and gender_model.input_shape[-1] == 18432:
        img = img.reshape(1, -1)

    return img

# Preprocess for Eye Model
def preprocess_eye_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        return redirect(url_for("home"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        file.seek(0)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        blood_image = preprocess_blood_image(file_bytes)
        blood_prediction = blood_model.predict(blood_image) if blood_model else None
        blood_result = BLOOD_CLASSES[np.argmax(blood_prediction)] if blood_prediction is not None else "Unknown"

        gender_image = preprocess_gender_image(file_path)
        gender_prediction = gender_model.predict(gender_image)[0][0] if gender_model else None
        male_prob = float(gender_prediction) if gender_prediction is not None else 0
        female_prob = 1 - male_prob if gender_prediction is not None else 0
        gender_result = "Male" if male_prob > female_prob else "Female"

        return render_template(
            "result.html",
            filename=filename,
            blood_group=blood_result,
            gender=gender_result,
            male_prob=f"{male_prob*100:.2f}",
            female_prob=f"{female_prob*100:.2f}",
            eye_result=None
        )

    except Exception as e:
        return str(e), 500

@app.route("/capture", methods=["POST"])
def capture():
    image_data = request.form["image_data"]

    header, encoded = image_data.split(",", 1)
    decoded = base64.b64decode(encoded)

    image_path = os.path.join(app.config["CAMERA_FOLDER"], "captured_eye.jpg")
    with open(image_path, "wb") as f:
        f.write(decoded)

    image = cv2.imread(image_path)
    eye_image = preprocess_eye_image(image)
    eye_prediction = eye_model.predict(eye_image) if eye_model else None
    eye_result = EYE_CLASSES[np.argmax(eye_prediction)] if eye_prediction is not None else "Unknown"

    return render_template(
        "result.html",
        eye_result=eye_result,
        blood_group=None,
        gender=None,
        male_prob=None,
        female_prob=None,
        filename=None
    )

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return redirect(f"/static/uploads/{filename}")

@app.route("/static/camera/<filename>")
def camera_file(filename):
    return redirect(f"/static/camera/{filename}")

if __name__ == "__main__":
    app.run(debug=True)
