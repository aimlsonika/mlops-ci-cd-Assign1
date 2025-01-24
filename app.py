"""
This is flask application
"""
#Importing the Library
import pandas as pd
from flask import request
from flask import Flask, jsonify
import joblib
from sklearn.utils import validation

# Initialize Flask application
app = Flask(__name__)
validation._IS_DEPRECATED_PICKLE = True  # This suppresses the warning
model = joblib.load('mlops_ci_cd_assign1/modelfinal_best.joblib')
@app.route('/diabetes_prediction', methods=['POST'])  
def diabetes_prediction():
            data = request.json
            df = pd.DataFrame(data["data"])
            #Drop if any features not required
            #df = df.drop(['Age','TemplateID'],axis=1)
            # Dataframe Creation
            df = df[['Pregnancies','Glucose','BloodPressure','SkinThickness',\
                'Insulin','BMI','DiabetesPedigreeFunction','Age']]
            df.Pregnancies 				= df.Pregnancies.astype('int64')              
            df.Glucose 					= df.Glucose.astype('int64')
            df.BloodPressure     		= df.BloodPressure.astype('int64')
            df.SkinThickness    		= df.SkinThickness.astype('int64')
            df.Insulin    				= df.Insulin.astype('int64')
            df.BMI  					= df.BMI.astype('float64')
            df.DiabetesPedigreeFunction = df.DiabetesPedigreeFunction.astype('float64')
            df.Age   					= df.Age.astype('int64')
            input = df.iloc[:,:]
            output = model.predict(input)
            final_predictions = pd.DataFrame(list(output),columns = \
                ["Based on your test report, the diabetes result is:"])\
                    .to_dict(orient="records")
            return jsonify(final_predictions)
app.run(debug=False,host = "127.0.0.1", port = 5000)  