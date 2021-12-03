from flask import Flask, request, render_template, redirect, url_for
import os

from medNet import MedNet
import predict_utils

PATH = os.getcwd()


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist("image")
        image_list = []
        for file in files:
            image_list.append(predict_utils.image_transform(file))
        return render_template("index.html",image_list = image_list)
    else:
        return render_template("index.html", zip = zip)

if __name__ == "__main__":
    app.run(debug=True)