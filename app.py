from flask import Flask, render_template, url_for, request  
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('i.html')

@app.route('/countcells', methods=['POST'])
def countCells():
    fileStr = request.files["img"].read()
    npImg = np.frombuffer(fileStr, np.uint8)
    img = cv2.imdecode(npImg, cv2.IMREAD_UNCHANGED)
    cv2.imshow('Imagen desplayada del server.',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "<h1>Correctamente enviada al server</h1>"

if __name__ == "__main__":
    app.run(debug=True)