from flask import Flask, render_template, url_for, request  
from jinja2 import Template
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
    template = Template("<h3>Nro de Globulos rojos {{ nrojos }}</h3></br> <h3>Nro de Leucositos {{ nleucositos }}</h3></br> <img src= {{ img }}>")
    return template.render(nrojos=str(count[0]), nleucositos=str(count[1]), img="\"static\img\processedImage.png\"") # This obviously wont work but is a try.

if __name__ == "__main__":
    app.run(debug=True)