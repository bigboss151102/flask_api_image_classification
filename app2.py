from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
from flasgger import Swagger
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
import os
import uuid
from PIL import Image
from io import BytesIO

app = Flask(__name__)
swagger = Swagger(app)
# Tạo một thư mục để lưu trữ ảnh
UPLOAD_FOLDER = "./images/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = ResNet50()


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


def is_valid_image(data):
    try:
        image = Image.open(BytesIO(data))
        image.verify()
        return True
    except Exception as e:
        print("Invalid image data:", str(e))
        return False


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu nhị phân từ request
        image_data = request.data

        if not is_valid_image(image_data):
            response = {"status": "error", "message": "Invalid image data"}
            return jsonify(response), 400
        # Tạo một tên tệp ngẫu nhiên
        image_filename = str(uuid.uuid4()) + ".jpg"

        # Tạo đường dẫn đầy đủ tới tệp ảnh
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Lưu trữ dữ liệu nhị phân của ảnh vào tệp
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)

        # Xử lý ảnh và dự đoán
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        label = decode_predictions(yhat)
        label = label[0][0]

        classification = {'class': label[1], 'confidence': label[2] * 100}

        result_text = f"Class: {classification['class']}, Confidence: {classification['confidence']}"
        response = make_response(result_text)
        response.headers["Content-Type"] = "text/plain"
        return response
    except Exception as e:
        print("Error:", str(e))
        response = {"status": "error", "message": "Error processing image"}
        return jsonify(response)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
