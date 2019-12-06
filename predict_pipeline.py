import numpy as np
import pandas as pd
import joblib
from utils import PipelineMixin


class PredictPipeline(PipelineMixin):
    """Pipeline to predict the Attrition class based on the existing model."""

    def __init__(self, scaler_name, model_name):
        self.scaler_name = scaler_name
        self.model_name = model_name


    def scale_data(self, input_data):
        self.scaler = joblib.load(self.scaler_name)
        input_data = pd.DataFrame(self.scaler.transform(input_data), columns=input_data.columns)
        return input_data


    def preprocess(self, input_data):
        input_data = self.select_top_features(input_data, y=False)
        input_data = self.apply_log_numerical_vars(input_data)
        input_data = self.get_dummies(input_data, y=False)
        input_data = self.scale_data(input_data)
        return input_data
    

    def predict(self, input_data):
        input_data = self.preprocess(input_data)
        self.model = joblib.load(self.model_name)
        prediction = self.model.predict(input_data)[0]
        probability = round(self.model.predict_proba(input_data)[0][1], 2)
        return prediction, probability


if __name__ == "__main__":
    input_data = {'Education': 4, 'NumCompaniesWorked': 7, 'MaritalStatus': 'Divorced', 'TotalWorkingYears': 10, 'BusinessTravel': 'Travel_Rarely', 'YearsSinceLastPromotion': 2, 'YearsAtCompany': 8, 'JobSatisfaction': 1, 'DistanceFromHome': 1, 'YearsWithCurrManager': 0, 'MonthlyIncome': 10000, 'EnvironmentSatisfaction': 1, 'OverTime': 'Yes'}
    input_data = pd.DataFrame.from_records([input_data])
    pipeline = PredictPipeline(scaler_name='scaler.pkl', model_name='logreg_model.pkl')
    prediction = pipeline.predict(input_data)
    print(prediction)

    