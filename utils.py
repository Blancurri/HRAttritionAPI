import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PipelineMixin:

    def select_top_features(self, data, y=True):
        top_features = ['Education', 'MaritalStatus', 'NumCompaniesWorked', 'TotalWorkingYears',
            'BusinessTravel', 'DistanceFromHome', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction', 'OverTime']
        if y:
            top_features.append('Attrition')
        return data[top_features]


    def apply_log_numerical_vars(self, data):
        columns_to_transform = ['TotalWorkingYears', 'DistanceFromHome', 'YearsSinceLastPromotion', 
                'YearsWithCurrManager', 'YearsAtCompany', 'MonthlyIncome']
        for column in columns_to_transform:
            if min(data[column]) == 0: 
                data[column] = np.log(np.add(data[column], 1))
            else:
                data[column] = np.log(data[column])
        return data


    def get_dummies(self, data, y=True):
        if y:
            data.loc[:, 'Attrition'] = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
        data.loc[:, 'OverTime'] = data['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
        data['MaritalStatus'] = pd.Categorical(data['MaritalStatus'], categories=['Single', 'Married', 'Divorced']) 
        data['BusinessTravel'] = pd.Categorical(data['MaritalStatus'], categories=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']) 
        data = pd.get_dummies(data, drop_first=True)
        return data


    def split_target_variable(self, data):
        y = data['Attrition']
        X = data.drop('Attrition', axis=1) 
        return X, y


    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test
