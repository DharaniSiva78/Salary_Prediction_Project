from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def get_preprocessor():
    numeric_features = [
        'Total_Experience', 'Total_Experience_in_field_applied',
        'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD',
        'Current_CTC', 'Inhand_Offer', 'Last_Appraisal_Rating',
        'No_Of_Companies_worked', 'Number_of_Publications', 'Certifications'
    ]

    categorical_features = [
        'Department', 'Role', 'Industry', 'Organization', 'Designation',
        'Education', 'Graduation_Specialization', 'University_Grad',
        'PG_Specialization', 'University_PG', 'PHD_Specialization',
        'University_PHD', 'Curent_Location', 'Preferred_location',
        'International_degree_any'
    ]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor
