import pandas as pd
import joblib

model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
actdx_mapping = joblib.load("actdx_mapping.pkl")

test_df = pd.read_csv("data/test_data.csv")

ordinal_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
binary_mapping = {"N": 0, "Y": 1}

binary_features = ['FH of Thyroid', 'FH of Cancer', 'FH of Diabetes', 'Married', 'Weight Gain',
                    'Weight Loss (Yes/No)', 'ALT / AST ()', 'Any antibodies present in blood',
                      'History of smoking', 'Anxiety', 'Irritability', 'Palpitations',
                        'Increased appetite', 'Muscle weakness', 'Tremors', 'Increased sweating',
                          'Heat intolerance', 'Dry skin/nails', 'Eyes bulging', 'Dry eyes', 'Fatigue',
                            'Fast Heart Rate', 'Slow heart rate', 'Decreased appetite', 'Cold intolerance',
                              'Nausea', 'Enlarged thyroid', 'Hoarseness', 'Swallowing issues', 'Hair Thinning',
                                'Autoimmune disease', 'Change in Bowel Movement', 'Irregular Periods', 'Joint Stiffness']

ordinal_features = ['Alcohol', 'Exercise']

for binary_feature in binary_features:
    test_df[binary_feature] = test_df[binary_feature].map(binary_mapping)

for ordinal_feature in ordinal_features:
    test_df[ordinal_feature] = test_df[ordinal_feature].map(ordinal_mapping)

test_df[['Systolic', 'Diastolic']] = test_df['BP (mmHg)'].str.split('/', expand=True).astype(int)

# Drop columns that were not used in training
test_df = test_df.drop(columns=["BP (mmHg)", "Name", "Occupation"], errors="ignore")

X_test_data = test_df[feature_columns]  

predictions = model.predict(X_test_data)
decoded_predictions = actdx_mapping[predictions]

for idx, prediction in enumerate(decoded_predictions):
    print(f"Prediction for {test_df['ID'].iloc[idx]}: {prediction}")
