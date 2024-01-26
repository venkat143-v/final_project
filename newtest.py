import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle
import sys
import time

with open(r'newf.json') as file:
    data = json.load(file)


def chat():
     # load trained model
    model = keras.models.load_model('newf_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        #print(tag)
        #verbs=['do', 'had', 'which','available?', 'why','about','aboiut','way', 'there','valid', 'us??','for','how','was', 'want', 'you', 'process', 'respective','anyone', 'as', 'and','explain', 'all', 'any','tell', 'many', 'if', 'Did', 'better', 'do?', 'provides', 'available?''classes?','t','Who','sir','this', 'use','meant?','does','prefer', 'what','with','details?','Can', 'gives']
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChaT:" + Style.RESET_ALL,end="")
                if inp.lower() in i['patterns']:
                    for j in np.random.choice(i['responses'])+'\n':
                        sys.stdout.write(j)
                        sys.stdout.flush()
                        time.sleep(random.random()*0.01)
                else:
                    for j in "Please check the spellings in your question.If the spellings are correct then the question is not related to this bot"+'\n':
                        sys.stdout.write(j)
                        sys.stdout.flush()
                        time.sleep(random.random()*0.01)

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
