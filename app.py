from flask import Flask, request, jsonify
import joblib

# Load the trained pipeline
pipeline = joblib.load('trained_pipeline.pkl')  # Replace 'trained_pipeline.pkl' with the path to your trained pipeline file

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
  # Check if request has content
  if not request.is_json:
    return jsonify({'error': 'Request must be JSON'}), 400

  # Get content from request body
  content = request.get_json()['content']

  # Make prediction
  prediction = predict_extremism(content)

  # Return prediction as JSON
  return jsonify({'prediction': prediction})

def predict_extremism(content):
    # Predict whether the content is extremist or not
    prediction = pipeline.predict([content])[0]
    return 'Extremist' if prediction == 1 else 'Not Extremist'

if __name__ == '__main__':
    app.run(debug=True)
