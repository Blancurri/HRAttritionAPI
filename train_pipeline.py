import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from utils import PipelineMixin
import warnings
warnings.filterwarnings('ignore')


class TrainPipeline(PipelineMixin):
    """Pipeline to train the model."""

    def __init__(self, train_data):
        self.data = train_data


    def __repr__(self):
        return f"<model_namePipeline: {self.model_name}>"
    

    def scale_data(self):
        columns = self.X.columns
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self.X_train = pd.DataFrame(self.scaler.transform(self.X_train), columns=columns)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=columns)
        joblib.dump(self.scaler, 'scaler.pkl')


    def preprocess(self, input_data=None):
        self.data = self.select_top_features(self.data)
        self.data = self.apply_log_numerical_vars(self.data)
        self.data = self.get_dummies(self.data)
        self.X, self.y = self.split_target_variable(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(self.X, self.y)
        self.scale_data()


    def fit(self):
        self.preprocess()
        self.model = LogisticRegression(class_weight='balanced', random_state=0)
        self.model.fit(self.X_train, self.y_train)
        joblib.dump(self.model, 'logreg_model.pkl')



if __name__ == "__main__":
    print("Running Pipeline")

    data = pd.read_csv('HR_DS.csv')
    input_data = data.iloc[[0], :]

    # Use Case #1:
    # init pipeline with train_data, train scaler and model
    # must run fit()
    pipeline = TrainPipeline(train_data=data)
    pipeline.fit()
