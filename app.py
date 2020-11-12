from flask import Flask

a == Flask(__name__)


@app.route('/')
def index():
    return "Hello, world!"
