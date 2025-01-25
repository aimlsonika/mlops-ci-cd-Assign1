"""
This is a Flask application for diabetes prediction.
"""

# Standard library imports
import warnings

# Third-party imports
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", message=".*_IS_DEPRECATED_PICKLE.*")

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('mlops_ci_cd_assign1/modelfinal_best.joblib')


@app.route('/diabetes_prediction', methods=['POST'])
def diabetes_prediction():
    """
    Endpoint for predicting diabetes based on user input.
    Expects JSON input with a 'data' key containing a list of records.
    """
    try:
        # Parse input JSON
        data = request.json
        if "data" not in data:
            return jsonify({"error": "Invalid input format. 'data' key missing."}), 400

        # Create DataFrame from input data
        df = pd.DataFrame(data["data"])

        # Ensure required features are present
        required_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns. Expected {required_columns}"}), 400

        # Convert columns to appropriate data types
        df = df.astype({
            'Pregnancies': 'int64',
            'Glucose': 'int64',
            'BloodPressure': 'int64',
            'SkinThickness': 'int64',
            'Insulin': 'int64',
            'BMI': 'float64',
            'DiabetesPedigreeFunction': 'float64',
            'Age': 'int64',
        })

        # Make predictions
        user_input = df.iloc[:, :]
        model_output = model.predict(user_input)
        final_predictions = pd.DataFrame(
            model_output, columns=["Based on your test report, the diabetes result is:"]
        ).to_dict(orient="records")

        # Return predictions as JSON
        return jsonify(final_predictions)

    except ValueError as value_error:
        return jsonify({"error": f"Data type conversion error: {str(value_error)}"}), 400

    except KeyError as key_error:
        return jsonify({"error": f"Missing key in input: {str(key_error)}"}), 400

    except Exception as general_error:  # Retained as a last resort
        return jsonify({"error": f"An unexpected error occurred: {str(general_error)}"}), 500


if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=5000)
