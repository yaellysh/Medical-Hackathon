from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/train_data.csv")

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

ordianl_features = ['Alcohol', 'Exercise']

for binary_feature in binary_features:
    df[binary_feature] = df[binary_feature].map(binary_mapping)

for ordinal_feature in ordianl_features:
    df[ordinal_feature] = df[ordinal_feature].map(ordinal_mapping)

df = df.drop(columns=["Name", "Occupation"])
df[['Systolic', 'Diastolic']] = df['BP (mmHg)'].str.split('/', expand=True).astype(int)
df = df.drop(columns=["BP (mmHg)"])

df['Misdiagnosis'], misdx_mapping = pd.factorize(df['Misdiagnosis'])
df['Actual Diagnosis'], actdx_mapping = pd.factorize(df['Actual Diagnosis'])


X = df.drop(columns=["Misdiagnosis", "Actual Diagnosis"])
y = df["Misdiagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
decoded_predictions = [actdx_mapping[i] for i in predictions]

print(decoded_predictions)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

test_df = pd.read_csv("data/test_data.csv")

for binary_feature in binary_features:
    test_df[binary_feature] = test_df[binary_feature].map(binary_mapping)

for ordinal_feature in ordianl_features:
    test_df[ordinal_feature] = test_df[ordinal_feature].map(ordinal_mapping)

test_df[['Systolic', 'Diastolic']] = test_df['BP (mmHg)'].str.split('/', expand=True).astype(int)

test_df = test_df.drop(columns=["Name", "Occupation"])

# Ensure the test data has the same columns as the training data
X_test_data = test_df[X_train.columns]

predictions = model.predict(X_test_data)

decoded_predictions = actdx_mapping[predictions]

test_names = test_df["ID"]
for idx, prediction in enumerate(decoded_predictions):
    print(f"Prediction for {test_names.iloc[idx]}: {prediction}")


