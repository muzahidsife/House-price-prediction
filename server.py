from flask import Flask, render_template, request, jsonify
import pickle
import json

app = Flask(__name__)

model_file_path = r'd:\Python\House price prediction\bangalore_home_prices_model.pickle'
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)



# Load the column names
with open("columns.json", "r") as f:
    columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the request
        data = request.get_json()

        # Ensure the input keys match the column names used during training
        features = [data[col.lower()] for col in columns['data_columns']]

        # Make the prediction
        prediction = model.predict([features])

        # Return the prediction as JSON
        return jsonify({'price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
