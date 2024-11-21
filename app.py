from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import numpy as np
import pickle
import requests

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  

API_KEY = 'fd25ea0e119a81e4ff531957e22c5f84'  
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

@app.route('/')
def page1():
    return render_template("page1.html")

@app.route('/page2')
def page2():
    return render_template("page2.html")

@app.route('/index')
def index():
    top_crops = session.pop('top_crops', None)
    weather_data = session.pop('weather_data', None)
    error_message = session.pop('error', None)
    return render_template("index.html", top_crops=top_crops, weather_data=weather_data, error=error_message)


@app.route('/values', methods=['POST'])
def values():
    try:
        # Save user inputs into the session
        session['N'] = float(request.form['nitrogen'])
        session['P'] = float(request.form['phosphorous'])
        session['K'] = float(request.form['potassium'])
        session['pH'] = float(request.form['ph1'])
        return redirect(url_for('index'))
    except ValueError:
        session['error'] = "Invalid input values. Please enter valid numbers."
        return redirect(url_for('index'))


@app.route("/predict", methods=['POST'])
def predict():
    try:
        N = session.get('N', 0)
        P = session.get('P', 0)
        K = session.get('K', 0)
        pH = session.get('pH', 0)
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, pH, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        probabilities = model.predict_proba(sc_mx_features)[0]

        sorted_indices = np.argsort(probabilities)[::-1]
        recommended_crops = [
            {"name": crop_dict[idx + 1], "image": f"images/{crop_dict[idx + 1]}.png"}
            for idx in sorted_indices if probabilities[idx] > 0
        ]
        session['top_crops'] = recommended_crops
        return redirect(url_for('index'))

    except Exception as e:
        session['error'] = str(e)
        return redirect(url_for('index'))


@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        city = request.form['City']

        url = f'{BASE_URL}?q={city}&appid={API_KEY}&units=metric'
        response = requests.get(url)
        data = response.json()

        if data['cod'] == 200:
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            rainfall = data.get('rain', {}).get('1h', 0)

            session['weather_data'] = {
                'temperature': temperature,
                'humidity': humidity,
                'rainfall': rainfall
            }
            return redirect(url_for('index'))
        else:
            session['error'] = 'City not found or weather data unavailable.'
            return redirect(url_for('index'))

    except Exception as e:
        session['error'] = str(e)
        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)