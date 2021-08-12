from joblib import load

model = load('./savedModels/model.joblib')
tfidf = load('./savedModels/tfidf.joblib')

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods = ['POST'])
@cross_origin()
def basic():
    text = request.json['text']
    X_test = tfidf.transform([text]).toarray()
    y_pred = model.predict(X_test)
    return jsonify({'response' : y_pred[0]})

if __name__ == '__main__':
    app.run(debug = True)