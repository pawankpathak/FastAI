from flask import Flask,render_template,flash, request, redirect
import pickle
from fastai.vision import *

application= app = Flask(__name__)

learn = pickle.load(open('./model/finalized_model.sav', 'rb'))

@app.route('/',methods =['GET' , 'POST'])
def original():
    if request.method == "POST":
        if request.files:
         image = request.files["image"]
         pred_class=classifyImage(image)
         print(pred_class)
         return render_template('index.html', pred_class= pred_class)
    return render_template('index.html')


def classifyImage(path):
 img = open_image(path)
 pred_class, pred_idx, output = learn.predict(img)
 return pred_class

if __name__ == '__main__':
 app.run(debug=True)
