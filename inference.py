import h2o
from flask import Flask, request, jsonify

# initialize an H2O cluster
h2o.init()

# load the saved model from the file path
model_path = './XGBoost_3_AutoML_1_20230410_114821'

#model_path = '/path/to/saved/model'
model = h2o.load_model(model_path)

# create a Flask app
app = Flask(__name__)

# create an endpoint for receiving new data and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get the request data as JSON and create an H2OFrame
    request_data = request.json
    new_data = h2o.H2OFrame(request_data)

    # use the loaded model to make predictions on the new data
    predictions = model.predict(new_data)

    # convert the H2OFrame to a Pandas DataFrame and then to JSON
    predictions_df = predictions.as_data_frame()
    predictions_json = predictions_df.to_json(orient='records')

    # return the predicted values as JSON
    return jsonify(predictions_json)

if __name__ == '__main__':
    # run the Flask app
    app.run(debug=True,port=5003)
