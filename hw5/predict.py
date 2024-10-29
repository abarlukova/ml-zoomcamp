import pickle
from flask import Flask, request, jsonify



with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)



with open('model1.bin', 'rb') as f_in:
    model = pickle.load(f_in)


app = Flask('subscription')


#client = {"job": "management", "duration": 400, "poutcome": "success"}


#print('input', client)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'subscription_proba' : float(y_pred)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

#print('client score', score)





