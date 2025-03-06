import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("myopia.csv", sep=';')

train, test = np.split(df.sample(frac=1, random_state=42), [int(0.8 * len(df))])

x_train = train.drop(columns=["MYOPIC", "ID", "STUDYYEAR"])
y_train = train["MYOPIC"].values

x_test = test.drop(columns=["MYOPIC", "ID", "STUDYYEAR"])
y_test = test["MYOPIC"].values

categorical_cols = ["GENDER"]
numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),  
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),  
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)), 
    ]
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)

print("\nModel Evaluation:\n", classification_report(y_test, y_pred))

print("\nEnter values for prediction:")
user_input = []
for col in x_train.columns:
    if col in categorical_cols:
        value = input(f"{col} ({df[col].unique()}): ")
    else:
        value = float(input(f"{col}: "))
    user_input.append(value)

user_input_df = pd.DataFrame([user_input], columns=x_train.columns)

probabilities = model.predict_proba(user_input_df)

prob_myopic = probabilities[0][1] * 100
prob_not_myopic = probabilities[0][0] * 100

print("\nPrediction Probabilities (in %):")
print(f"Myopic: {prob_myopic:.2f}%")
print(f"Not Myopic: {prob_not_myopic:.2f}%")

final_prediction = "Myopic" if prob_myopic > prob_not_myopic else "Not Myopic"
print(f"\nFinal Prediction: {final_prediction}")