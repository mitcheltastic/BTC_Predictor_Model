# app.py
from flask import Flask, jsonify
from predictor_api import make_prediction # <-- Re-using your existing logic

# Initialize the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    """
    This function runs the prediction and returns it as JSON.
    """
    try:
        # Get the prediction result from your original function
        result = make_prediction()

        # If your prediction logic returns an error, pass it along
        if "error" in result:
            return jsonify(result), 400 # Bad Request

        # Return the successful result as JSON
        return jsonify(result)

    except Exception as e:
        # Handle any unexpected errors during prediction
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Runs the app on http://127.0.0.1:5000
    app.run(debug=True)