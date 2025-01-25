"""
This is a Flask application for predicting diabetes based on input data.
"""
import logging
import warnings
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from sklearn.utils import validation

# Suppress warnings and configure logging
# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", message=".*_IS_DEPRECATED_PICKLE.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load("modelfinal_best.joblib")
    logging.info("Model successfully loaded.")
except FileNotFoundError:
    logging.error("Model file not found. Please ensure the model file is in the specified path.")
    raise
except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
    logging.error("Error loading model: %s",e)
    raise

@app.route('/diabetes_prediction', methods=['POST'])
def diabetes_prediction():
    """
    Endpoint to predict diabetes based on user input.
    """
    try:
        # Parse input data
        data = request.json
        if not data or "data" not in data:
            return jsonify({"error": "Invalid input. \
                            Please provide data in JSON format."}), 400
        # Create a DataFrame from the input data
        df = pd.DataFrame(data["data"])

        # Check if required columns are present
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(column in df.columns for column in required_columns):
            return jsonify({"error": "Missing required columns. \
                            Expected: %s"},required_columns), 400
        # Convert data types to match the model's requirements
        df = df[required_columns]
        df = df.astype({
            'Pregnancies': 'int64',
            'Glucose': 'int64',
            'BloodPressure': 'int64',
            'SkinThickness': 'int64',
            'Insulin': 'int64',
            'BMI': 'float64',
            'DiabetesPedigreeFunction': 'float64',
            'Age': 'int64'
        })

        # Make predictions
        predictions = model.predict(df)
        final_predictions = pd.DataFrame(
            list(predictions),
            columns=["Based on your test report, the diabetes result is:"]
        ).to_dict(orient="records")

        return jsonify(final_predictions), 200

    except pd.errors.EmptyDataError:
        logging.error("Received empty data.")
        return jsonify({"error": "Input data is empty."}), 400
    except ValueError as e:
        logging.error("Value error during prediction: %s",e)
        return jsonify({"error": "Invalid data provided: %s"},e), 400
    except Exception as e:
        logging.error("Unexpected error: %s",e)
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    try:
        logging.info("Starting Flask app...")
        app.run(debug=False, host="127.0.0.1", port=5000)
    except Exception as e:
        logging.error("Failed to start Flask app: %s",e)
        raise

'''

"""
This is a Flask application for diabetes prediction.
"""

# Standard library imports
import warnings

# Third-party imports
import pandas as pd
from flask import Flask, request, jsonify
import joblib



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

    except pd.errors.EmptyDataError:
        return jsonify({"error": "The input data is empty or invalid."}), 400

    except joblib.externals.loky.process_executor.BrokenProcessPool as model_error:
        return jsonify({"error": f"Model loading or prediction failed: {str(model_error)}"}), 500

    except Exception as general_error:
        # Use as a last resort and provide a justification
        return jsonify({"error": f"An unexpected error occurred: {str(general_error)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=5000)
'''
