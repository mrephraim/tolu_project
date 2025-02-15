import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier  # For multi-output predictions
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # Handling class imbalance
import joblib

# Load dataset
path = "s.csv"
df = pd.read_csv(path)

# Data Preprocessing
df.drop_duplicates(inplace=True)  # Remove duplicates

# Convert Yes/No columns (except AGE)
def encode_binary_columns(df):
    for column in df.columns:
        if column != "AGE":  # Keep AGE numeric
            df[column] = df[column].replace({2: 1, 1: 0})  # Convert 2 → 1 (Yes), 1 → 0 (No)
    return df

df = encode_binary_columns(df)

# Convert categorical values to numeric
df["GENDER"] = df["GENDER"].map({"M": 1, "F": 0})  # Convert GENDER to 1 (M) and 0 (F)
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})  # Convert LUNG_CANCER to binary

# Assign Biomarkers Randomly for Patients with Lung Cancer (1-5)
df.loc[df["LUNG_CANCER"] == 1, "BIOMARKER"] = np.random.choice([1, 2, 3, 4, 5], 
                                                                size=df["LUNG_CANCER"].sum(), replace=True)

df["BIOMARKER"] = df["BIOMARKER"].fillna(0)  # Safe way

# Feature Selection
X = df.drop(columns=["LUNG_CANCER", "BIOMARKER"])  # Features
Y = df[["LUNG_CANCER", "BIOMARKER"]]  # Multi-output Target

# Handle class imbalance using SMOTE (Only on LUNG_CANCER, not Biomarker)
smote = SMOTE(random_state=42)
X_resampled, Y_resampled_cancer = smote.fit_resample(X, Y["LUNG_CANCER"])

# Re-assign biomarkers to synthetic lung cancer cases
Y_resampled = pd.DataFrame({"LUNG_CANCER": Y_resampled_cancer})
Y_resampled.loc[Y_resampled["LUNG_CANCER"] == 1, "BIOMARKER"] = np.random.choice(
    [1, 2, 3, 4, 5], size=Y_resampled["LUNG_CANCER"].sum(), replace=True
)
Y_resampled["BIOMARKER"] = Y_resampled["BIOMARKER"].fillna(0)  # Fill non-cancer cases with 0

# Convert BIOMARKER to categorical
le = LabelEncoder()
Y_resampled["BIOMARKER"] = le.fit_transform(Y_resampled["BIOMARKER"])

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=40)

# Standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Separate Models for Multi-Output Learning
cancer_model = LogisticRegression()
biomarker_model = RandomForestClassifier(n_estimators=100)

# Train Cancer Model
cancer_model.fit(x_train, y_train["LUNG_CANCER"])

# Train Biomarker Model (only on cancer cases)
biomarker_model.fit(x_train[y_train["LUNG_CANCER"] == 1], y_train[y_train["LUNG_CANCER"] == 1]["BIOMARKER"])

# Predictions
cancer_pred = cancer_model.predict(x_test)
biomarker_pred = np.zeros(len(cancer_pred))  # Default to 0 (No biomarker)

# Only predict biomarkers if lung cancer is detected
cancer_cases = (cancer_pred == 1)
biomarker_pred[cancer_cases] = biomarker_model.predict(x_test[cancer_cases])

# Evaluate Models
print("Classification Report - Lung Cancer:")
print(classification_report(y_test["LUNG_CANCER"], cancer_pred))

print("Classification Report - Biomarker:")
print(classification_report(y_test["BIOMARKER"], biomarker_pred))

# Save Models and Scaler
joblib.dump(cancer_model, "cancer_model.joblib")
joblib.dump(biomarker_model, "biomarker_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")

print("Models saved successfully!")

# Load the trained models and scaler
cancer_model = joblib.load("cancer_model.joblib")
biomarker_model = joblib.load("biomarker_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")

# Function for user input and prediction
def predict_lung_cancer():
    print("Enter the required features for prediction:")
    user_data = []
    feature_names = X.columns.tolist()
    
    for feature in feature_names:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                user_data.append(value)
                break
            except ValueError:
                print("Invalid input! Please enter a valid number.")

    user_data = pd.DataFrame([user_data], columns=X.columns)  # Ensure correct feature names
    user_data = scaler.transform(user_data)  # Standardize input

    cancer_prediction = cancer_model.predict(user_data)
    
    if cancer_prediction[0] == 1:
        biomarker_prediction = biomarker_model.predict(user_data)[0]
        biomarker = le.inverse_transform([int(biomarker_prediction)])[0]  # Decode biomarker label
    else:
        biomarker = "N/A"

    lung_cancer_status = "YES" if cancer_prediction[0] == 1 else "NO"

    print(f"Predicted Lung Cancer Status: {lung_cancer_status}\nPossible Associated Biomarker: {biomarker}")

# Call function to take user input
predict_lung_cancer()
