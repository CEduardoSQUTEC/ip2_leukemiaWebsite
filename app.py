from flask import Flask, render_template, url_for, request  
import numpy as np
import cv2
import countingCode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/countcells', methods=['POST'])
def countCells():
    fileStr = request.files["img"].read()
    npImg = np.frombuffer(fileStr, np.uint8)
    img = cv2.imdecode(npImg, cv2.IMREAD_UNCHANGED)
    count = countingCode.countingImage(img)
    return "<h1>Correctamente procesada " + str(count[0]) + " " + str(count[1]) + "</h1>"

if __name__ == "__main__":
    app.run(debug=True)