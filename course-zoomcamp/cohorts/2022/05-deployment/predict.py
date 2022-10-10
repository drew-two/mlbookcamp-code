import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

def score(client):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    decision = y_pred >= 0.5

    return y_pred, decision

app = Flask('credit_card_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred, decision = score(client)

    result = {
        'probability': float(y_pred),
        'decision': bool(decision)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)