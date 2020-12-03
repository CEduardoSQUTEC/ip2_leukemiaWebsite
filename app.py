from flask import Flask, render_template, url_for, request  
<<<<<<< HEAD
import numpy as np
import cv2
import countingCode
=======
import cv2
import numpy as np
>>>>>>> 0a3064d84225034ddf706087dcaf02d3b2d89fc8

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/countcells', methods=['POST'])
def countCells():
    fileStr = request.files["img"].read()
    npImg = np.frombuffer(fileStr, np.uint8)
    img = cv2.imdecode(npImg, cv2.IMREAD_UNCHANGED)
    countingCode.countingImage(img)
    # cv2.imshow('Imagen desplayada del server.', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return "<h1>Correctamente procesada</h1>"

if __name__ == "__main__":
    app.run(debug=True)