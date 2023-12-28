import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2 as cv
# Define a flask app
application = Flask(__name__)

dic = {0: "Apple_scab", 1: "Black_rot", 2: "apple_rust", 3: "healthy"}

model = load_model("GaborVGG16_TrainedModel.h5")


def model_predict(img_path, model):
    img = cv.imread(img_path, 0)
    img = cv.resize(img, (256, 256))
    gabor_1 = cv.getGaborKernel((18, 18), 1.5, np.pi / 4, 5.0, 1.5, 0, ktype=cv.CV_32F)
    filtered_img_1 = cv.filter2D(img, cv.CV_8UC3, gabor_1)
    img2 = cv.merge((filtered_img_1, filtered_img_1, filtered_img_1))
    gabor_2 = cv.getGaborKernel((18, 18), 1.5, np.pi / 4, 5.0, 1.5, 0, ktype=cv.CV_32F)
    filtered_img_2 = cv.filter2D(filtered_img_1, cv.CV_8UC3, gabor_2)
    cv.imwrite("gabor_image/image_leaf_1.png", filtered_img_2)
    image1 = image.load_img("gabor_image/image_leaf_1.png", target_size=(256, 256))
    img1 = np.array(image1) / 255.0
    img = np.expand_dims(img1, axis=0)
    p = model.predict(img)
    x = np.argmax(p)
    return dic[x]


@application.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    application.run(debug=True)
