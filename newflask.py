from flask import*
import os
import base64
import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama 
from colorama import Fore, Style, Back
import random
import pickle
import sys
import time
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import cv2
import speech_recognition as sr
import webbrowser 
import librarybookscheck as lbc 



app = Flask(__name__)

UPLOAD_FOLDER = 'static/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')

@app.route('/', methods=['POST'])
def process_message():
    if 'Msg' in request.form:
        # Text message input
        inp = request.form['Msg']
        with open(r'newf.json') as file:
            data = json.load(file)
        model = keras.models.load_model('newf_model')
        # return 'hii'
        
        # load tokenizer object
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        f=0
        for i in data['intents']:
            if i['tag'] == tag:
                f=1
                #res=np.random.choice(i['responses'])
                #break
                if inp.lower() in [s.lower() for s in i['patterns']]:
                    res=np.random.choice(i['responses'])
                    break
                else:
                    res=lbc.check(inp)
                    break
        return res
    elif 'Audio' in request.files:
        # Handle audio upload
        audio_file = request.files['Audio']
        res="audio file received venkat"
        return res

    elif 'Image' in request.files:
        # Image upload input
        file = request.files['Image']
        # Save the image to the server
        na=file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        
        with open("emotion_model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('emotion_model.h5')

        # Compile the loaded model (necessary for predictions)
        loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        # Function to preprocess an image for prediction
        def preprocess_image(image_path):
            img = load_img(image_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            return img_array

        # Test image path
        test_image_path = r"static/data/"+na # Replace with the path of the test image

        # Preprocess the test image for prediction
        test_image = preprocess_image(test_image_path)

        # Make predictions on the test image
        prediction = loaded_model.predict(test_image)

        # Classify the prediction (assuming binary classification)
        os.remove(test_image_path)
        if(prediction[0][0]>=0.5):
            res="Dog"
        else:
            res="Cat"
        return res
    else:
        return render_template("indeximgaudtxtvoi.html")


if __name__ == '__main__':
    app.run(debug=True)

