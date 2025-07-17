# app.py
from flask import Flask, jsonify
# Import the function from your existing file
from predictor_api import make_prediction

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    """
    Runs the prediction and returns it as JSON.
    """
    try:
        result = make_prediction()
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

# Optional: Add a root endpoint for health checks
@app.route('/', methods=['GET'])
def health_check():
    return "API is running."

if __name__ == '__main__':
    app.run(debug=True)
