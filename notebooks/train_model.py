import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from src.data_preprocessing import get_preprocessor

df = pd.read_csv('../data/expected_ctc.csv')
df = df.dropna(subset=['Expected_CTC'])

numeric_columns = [
    'Total_Experience', 'Total_Experience_in_field_applied',
    'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD',
    'Current_CTC', 'Inhand_Offer', 'Last_Appraisal_Rating',
    'No_Of_Companies_worked', 'Number_of_Publications', 'Certifications'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid to NaN

y = df['Expected_CTC']
X = df.drop(['Expected_CTC', 'Applicant_ID', 'IDX'], axis=1)

preprocessor = get_preprocessor()
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

pipeline.fit(X, y)

os.makedirs('../models', exist_ok=True)

joblib.dump(pipeline, '../models/best_model.pkl')

print("✅ Model trained and saved to models/best_model.pkl")
