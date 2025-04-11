import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__, static_url_path='/static')  # Initialize the flask app

# Load the trained model and encoders
model = pickle.load(open('Model/trained_model.pkl', 'rb'))
feature_encoders = pickle.load(open('Model/feature_encoders.pkl', 'rb'))
label_encoder = pickle.load(open('Model/label_encoder.pkl', 'rb'))

@app.route('/')
@app.route('/index') 
def index():
    return render_template('index.html')

@app.route('/login') 
def login():
    return render_template('login.html')    

@app.route('/chart') 
def chart():
    return render_template('chart.html')    

@app.route('/abstract') 
def abstract():
    return render_template('abstract.html')    

@app.route('/performance') 
def performance():
    return render_template('performance.html')   

@app.route('/future') 
def future():
    return render_template('future.html')  

@app.route('/upload') 
def upload():
    return render_template('upload.html') 

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)    

@app.route('/home')
def home():
    return render_template('test.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]

        if len(features) != 9:
            return render_template('test.html', prediction_text="⚠️ Please enter all 9 input values.")

        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = label_encoder.inverse_transform(prediction)[0]

        return render_template('test.html', prediction_text=f"🛡️ Prediction: {output}")

    except Exception as e:
        return render_template('test.html', prediction_text=f"Error: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True)
