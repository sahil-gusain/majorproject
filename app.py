import os
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
from joblib import load
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing import image


app = Flask(__name__)


model =load_model('C:\\Users\\sahil\\Desktop\\majorProject\\CNN\\cnn_model.h5')
modelYolo = torch.load('C:\\Users\\sahil\\Desktop\\majorProject\\YOLO\\runs\\detect\\train\\weights\\best.pt')

modelsvm = load('C:\\Users\\sahil\Desktop\\majorProject\\SVM\\svm_models.dat')
modelKNNMAn=load('C:\\Users\\sahil\\Desktop\\majorProject\\KNN\\knn_manhattan.joblib')
modelKNNEu=load('C:\\Users\\sahil\\Desktop\\majorProject\\KNN\\knn_euclidean.joblib')

print('Model loaded. Check http://127.0.0.1:5000/')
class_label=['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Rotten Apple', 'Rotten Banana', 'Rotten Orange']


def get_className(classNo):
	return class_label[classNo]


def getResult(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'test', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)