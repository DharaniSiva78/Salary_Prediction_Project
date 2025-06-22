from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import joblib
import numpy as np

def train_model(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    model.fit(X, y)
    return model, np.mean(-scores)

def save_model(model, path):
    joblib.dump(model, path)
