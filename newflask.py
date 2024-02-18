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
from tensorflow.keras.optimizers import Adamz, Adamax
import cv2
import speech_recognition as sr
import webbrowser 
import librarybookscheck as lbc 
from tensorflow.keras.metrics import categorical_crossentropy
from PIL import Image


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
        
        def gif_to_jpg(gif_path,output_folder):
            gif = Image.open(gif_path)
            for frame_number in range(gif.n_frames):
                gif.seek(frame_number)
                frame = gif.convert('RGB')
                frame.save(f"{output_folder}\\modified.jpg")

        def png_to_jpg(img_path):
            im = Image.open(img_path) 
            rgb_im = im.convert("RGB") 
            rgb_im.save(r"E:\newfinal\static\data\modified.jpg")
            
        loaded_model = tf.keras.models.load_model('ec2Types.h5', compile=False)
        loaded_model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])
        
        # Test image path
        test_image_path = r"static/data/"+na # Replace with the path of the test image
        image = Image.open(test_image_path)
        
        file_extension = os.path.splitext(test_image_path)[-1].lower()

        if file_extension in [".png"]:
            png_to_jpg(test_image_path)
            image = Image.open(r"E:\newfinal\static\data\modified.jpg")
            
        elif file_extension in [".gif"]:
            gif_to_jpg(test_image_path, r"E:\newfinal\static\data")
            image = Image.open(r"E:\newfinal\static\data\modified.jpg")
        else:
            image = Image.open(test_image_path)
        # Preprocess the test image for prediction
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        classes=['Capacitor', 'IC', 'Resistor', 'Transistor']
        # Make predictions
        predictions = loaded_model.predict(img_array)
        class_labels = classes
        score = tf.nn.softmax(predictions[0])
        res1=f"{class_labels[tf.argmax(score)]}"
        if file_extension in ['.png','.gif']:
            os.remove(r"E:\newfinal\static\data\modified.jpg")
        os.remove(test_image_path)
        if(res1=="IC"):
            res="It is an IC. An integrated circuit (IC), sometimes called a chip, microchip or microelectronic circuit, is a semiconductor wafer on which thousands or millions of tiny resistors, capacitors, diodes and transistors are fabricated. An IC can function as an amplifier, oscillator, timer, counter, logic gate, computer memory, microcontroller or microprocessor."
        elif(res1=="Capacitor"):
            res="It is a Capacitor. A capacitor is a two-terminal electrical device that can store energy in the form of an electric charge. It consists of two electrical conductors that are separated by a distance.  The space between the conductors may be filled by vacuum or with an insulating material known as a dielectric. The ability of the capacitor to store charges is known as capacitance."
        elif(res1="Resistor"):
            res="It is a resistor. A passive electrical component with two terminals that are used for either limiting or regulating the flow of electric current in electrical circuits. It is made of copper wires which are coiled around a ceramic rod and the outer part of the resistor is coated with an insulating paint."
        else:
            res="It is a transistor. A transistor is a type of semiconductor device that can be used to conduct and insulate electric current or voltage. A transistor basically acts as a switch and an amplifier. In simple words, we can say that a transistor is a miniature device that is used to control or regulate the flow of electronic signals."
        return res
    else:
        return render_template("indeximgaudtxtvoi.html")


if __name__ == '__main__':
    app.run(debug=True)

