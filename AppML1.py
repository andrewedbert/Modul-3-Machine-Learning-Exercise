# flask w/ model ML
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/ml')
def ml():
    MedInc = 2.890600
    HouseAge = 16.000000
    AveRooms = 5.413043
    AveBedrms = 1.034161
    Population = 652.000000
    AveOccup = 2.024845
    Latitude = 32.850000
    Longitude = -116.890000 
    hasil = model.predict([[
        MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    ]])[0]
    return str(hasil)

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        body = request.json
        # 8 vars
        MedInc = body['MedInc']
        HouseAge = body['HouseAge']
        AveRooms = body['AveRooms']
        AveBedrms = body['AveBedrms']
        Population = body['Population']
        AveOccup = body['AveOccup']
        Latitude = body['Latitude']
        Longitude = body['Longitude']
        # model prediction
        hasil = model.predict([[
            MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        ]])[0]
        return jsonify({
            'MedInc' : body['MedInc'],
            'HouseAge' : body['HouseAge'],
            'AveRooms' : body['AveRooms'],
            'AveBedrms' : body['AveBedrms'],
            'Population' : body['Population'],
            'AveOccup' : body['AveOccup'],
            'Latitude' : body['Latitude'],
            'Longitude' : body['Longitude'],
            'prediksi' : hasil,
            'status' : 'Anda nge-POST'
        })
    elif request.method == 'GET':
        return jsonify({
            'status':'Anda nge-GET'
        })
    else:
        return jsonify({
            'status':'Anda tidak nge-POST & nge-GET'
        })



@app.route('/predictform', methods=['GET', 'POST'])
def predict1form():
    if request.method == 'POST':
        body = request.form
        MedInc = float(body['MedInc'])
        HouseAge = float(body['HouseAge'])
        AveRooms = float(body['AveRooms'])
        AveBedrms = float(body['AveBedrms'])
        Population = float(body['Population'])
        AveOccup = float(body['AveOccup'])
        Latitude = float(body['Latitude'])
        Longitude = float(body['Longitude'])
        hasil = model.predict([[
            MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
        ]])[0]
        return jsonify({
            'MedInc' : str(body['MedInc']),
            'HouseAge' : str(body['HouseAge']),
            'AveRooms' : str(body['AveRooms']),
            'AveBedrms' : str(body['AveBedrms']),
            'Population' : str(body['Population']),
            'AveOccup' : str(body['AveOccup']),
            'Latitude' : str(body['Latitude']),
            'Longitude' : str(body['Longitude']),
            'prediksi' : str(hasil),
            'status' : 'Anda nge-POST'
        })
        # return "ok"

if __name__ == '__main__':
    model = joblib.load('modelJoblib.joblib')
    app.run(debug = True)