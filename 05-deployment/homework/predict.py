import pickle
from flask import Flask, request, jsonify

input_dv = 'dv.bin'
input_model = 'model1.bin'

with open(input_dv, 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open(input_model, 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask('subscription')

def predict_subscription_proba(client, dv, model):
    X = dv.transform([client])
    probability = model.predict_proba(X)[0, 1]
    return probability

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    y_pred = predict_subscription_proba(client, dv, model)
    subscribe = y_pred >= 0.5

    result = {
        'subscribe_probability': float(y_pred),
        'subscribe': bool(subscribe)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
