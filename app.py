from flask import Flask, render_template, request
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
# from tensorflow.keras.applications.resnet50 import ResNet50

app = Flask(__name__)

model = load_model('./cat_dog_100epochs.h5')
print(model.summary())


@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/classifier', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image/255
    yhat = model.predict(image)
    print(yhat)
    if yhat > 0.5:
        classification = "Dog"
    else:
        classification = "Cat"

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
