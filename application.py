from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import sys
from src.Heart_Disease_Prediction.exception import CustomException
from src.Heart_Disease_Prediction.logger import logging

application = Flask(__name__)
app = application

# ✅ Model aur Preprocessor load karo
model_path = os.path.join('artifacts', 'model.pkl')
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

model = pickle.load(open(model_path, 'rb'))
preprocessor = pickle.load(open(preprocessor_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    
    try:
        # ✅ Form se data lo
        age      = float(request.form.get('age'))
        sex      = float(request.form.get('sex'))
        cp       = float(request.form.get('cp'))
        trestbps = float(request.form.get('trestbps'))
        chol     = float(request.form.get('chol'))
        fbs      = float(request.form.get('fbs'))
        restecg  = float(request.form.get('restecg'))
        thalach  = float(request.form.get('thalach'))
        exang    = float(request.form.get('exang'))
        oldpeak  = float(request.form.get('oldpeak'))
        slope    = float(request.form.get('slope'))
        ca       = float(request.form.get('ca'))
        thal     = float(request.form.get('thal'))

        # ✅ Array banao
        features = np.array([[age, sex, cp, trestbps, chol, 
                               fbs, restecg, thalach, exang, 
                               oldpeak, slope, ca, thal]])

        # ✅ Preprocess + Predict
        data_scaled = preprocessor.transform(features)
        prediction  = model.predict(data_scaled)
        result = "❤️ Heart Disease Detected!" if prediction[0] == 1 else "✅ No Heart Disease"

        return render_template('index.html', result=result)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)