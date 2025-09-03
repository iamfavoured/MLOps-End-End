import pickle
from flask import Flask, request, app, jsonify, url_for, render_template,redirect,flash
import numpy
import pandas


app = Flask(__name__)

## load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalmod = pickle.load(open('scaler.pkl','rb'))

##route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data = numpy.array(list(data.values())).reshape(1, -1)
    print(data)
    new_data = scalmod.transform(data)
    output = regmodel.predict(new_data)
    print(output)

    return jsonify(output[0])




if __name__ == "__main__":
    app.run(debug=True)
