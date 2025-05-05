from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        prediction = model.predict([np.array(data)])[0]
        result = "Positive (Has Diabetes)" if prediction == 1 else "Negative (No Diabetes)"
        return render_template('index.html', prediction_text=f'Result: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data["features"]])
    features = scaler.transform(features)
    features_tensor = torch.FloatTensor(features).to(device)
    
    # HiLe prediction
    with torch.no_grad():
        hile_pred = hile_model(features_tensor).squeeze().cpu().numpy()
        hile_prob = float(hile_pred)
    
    # HiTCLe prediction
    with torch.no_grad():
        base_preds = np.column_stack([
            model(features_tensor).squeeze().cpu().numpy() for model in base_models
        ])
        meta_pred = meta_model(torch.FloatTensor(base_preds).to(device)).squeeze().cpu().numpy()
        hitcle_prob = float(meta_pred)
    
    return jsonify({
        "hile_probability": hile_prob,
        "hile_prediction": int(hile_prob > 0.5),
        "hitcle_probability": hitcle_prob,
        "hitcle_prediction": int(hitcle_prob > 0.5)
    })

@app.route("/download", methods=["GET"])
def download_model():
    return send_file("model.tlx", as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)