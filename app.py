from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('i.html')

@app.route('/countcells', methods=['POST'])
def countCells():
    image = request.form['img']
    # return "<img src={} alt=\"Image\">".format(image)
    return "<h1>OK</h1>"

if __name__ == "__main__":
    app.run(debug=True)
