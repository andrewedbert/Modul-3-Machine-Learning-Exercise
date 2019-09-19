from flask import Flask, abort, render_template, redirect, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './file'

@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/file/<path:path>')
def akses(path):
    return send_from_directory('file',path)

@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        myfile = request.files['file']
        fn = secure_filename(myfile.filename)
        filex = np.random.rand(1)
        myfile.save(os.path.join(app.config['UPLOAD_FOLDER'], fn))
        dataDg = load_digits()
        xtr, xts, ytr, yts = train_test_split(
            dataDg['data'],
            dataDg['target'],
            test_size = .1
        )
        model = LogisticRegression(
            solver='lbfgs', 
            multi_class='auto',
            max_iter=10000
        )
        model.fit(xtr,ytr)
        gbr = Image.open(f'./file/{str(fn)}').convert('L')
        gbr = gbr.resize((8,8))
        gbr = PIL.ImageOps.invert(gbr)
        gbrArr = np.array(gbr)
        gbrArr2 = gbrArr.reshape(1,64)
        prediksi = model.predict(gbrArr2.reshape(1,-1))
        plt.imshow(gbrArr, cmap='gray')
        # plt.show()
        plt.title(
            f'P = {prediksi[0]} | D = {yts[0]}'
        )
        plt.savefig('./file/{}.png'.format(filex))
        # return str(prediksi)
        return render_template('res.html', data='{}.png'.format(filex))

if __name__ == '__main__':
    app.run(debug = True)