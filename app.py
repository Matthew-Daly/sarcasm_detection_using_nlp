from flask import Flask, render_template, request
import numpy as np
import keras.models
import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
from load import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
#init flask app
app = Flask(__name__)

global model, graph, tokenizer
model, graph = init()

tokenizer = pickle.load(open('tokenizer.pickle','rb'))



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        X = tokenizer.texts_to_sequences(data)
        X = pad_sequences(X, padding='post', maxlen=100)
        with graph.as_default():
            out = model.predict(X)
            out = out.item((0,0))   
    return render_template('result.html', prediction = out)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=5000)






