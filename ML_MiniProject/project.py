import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
import joblib

# Global variables
trained_model = None
scaler = None

# Function to preprocess data (handle categorical features, scaling, etc.)
def preprocess_data(df):
    global scaler
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
    
    # Convert categorical variables using label encoding
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Assuming "is_fraud" is the target column in your dataset
    X = df.drop('is_fraud', axis=1)  # Features (all except 'is_fraud')
    y = df['is_fraud']  # Target column

    # Scale numeric columns to ensure they are standardized
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to train the model
def train_model():
    global trained_model
    try:
        # Load dataset
        df = pd.read_csv("Book2.csv")

        # Clean the 'is_fraud' column
        df['is_fraud'] = df['is_fraud'].astype(str).str.strip()
        df['is_fraud'] = df['is_fraud'].replace({'1': 1, '0': 0})
        df['is_fraud'] = df['is_fraud'].astype(int)

        # Check the class distribution
        print("Class distribution in 'is_fraud' column after cleaning:")
        print(df['is_fraud'].value_counts())

        # Ensure there are at least two classes
        if df['is_fraud'].nunique() < 2:
            messagebox.showerror("Error", "The target variable 'is_fraud' needs at least two classes to train the model. Switching to anomaly detection.")
            train_anomaly_detection(df)  # Switch to anomaly detection if only one class
            return

        # Preprocess data
        X, y = preprocess_data(df)

        # Get model choice from user inputs
        model_choice = model_var.get()

        # Train the model based on user choice
        if model_choice == "LogisticRegression":
            trained_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif model_choice == "DecisionTree":
            trained_model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "RandomForest":
            trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        trained_model.fit(X, y)

        result_label.config(text="Model trained successfully. You can now use the model for predictions.")
        joblib.dump(trained_model, 'trained_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to train anomaly detection model if only one class exists
def train_anomaly_detection(df):
    global trained_model
    try:
        # Preprocess data (excluding the target variable)
        X, _ = preprocess_data(df)  # We won't use y since we have only one class

        # Train Isolation Forest model for anomaly detection
        trained_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        trained_model.fit(X)

        result_label.config(text="Anomaly detection model trained successfully. You can now use the model for predictions.")
        joblib.dump(trained_model, 'trained_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to make predictions
def make_prediction():
    global trained_model, scaler
    try:
        # Load the saved model and scaler
        trained_model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Input values
        amt = float(amt_entry.get())
        trans_num = int(trans_num_entry.get())

        # Create DataFrame for prediction with the same feature names and order used for training
        input_data = pd.DataFrame([[amt, trans_num]], columns=['amt', 'trans_num'])

        # Scale the input data using the trained scaler
        input_scaled = scaler.transform(input_data)

        # Make prediction based on the model type
        if isinstance(trained_model, IsolationForest):
            prediction = trained_model.predict(input_scaled)
            # Display result (-1 for anomaly, 1 for normal)
            if prediction[0] == -1:
                prediction_label.config(text="Prediction: Anomaly detected (Possible Fraud)", fg="red")
            else:
                prediction_label.config(text="Prediction: No anomaly detected (Normal)", fg="green")
        else:
            prediction = trained_model.predict(input_scaled)
            # Display result for classification models
            if prediction[0] == 1:
                prediction_label.config(text="Prediction: Fraud detected (Yes)", fg="red")
            else:
                prediction_label.config(text="Prediction: No fraud detected (No)", fg="green")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during prediction: {e}")

# Initialize GUI Window
root = Tk()
root.title("Fraud Detection Model Trainer")

# Create UI Elements
title_label = Label(root, text="Fraud Detection Model Trainer", font=("Helvetica", 16))
title_label.pack(pady=10)

model_label = Label(root, text="Select Model:")
model_label.pack(pady=5)

model_var = StringVar(root)
model_var.set("LogisticRegression")  # Default value

model_dropdown = OptionMenu(root, model_var, "LogisticRegression", "DecisionTree", "RandomForest")
model_dropdown.pack(pady=5)

train_btn = Button(root, text="Train Model", command=train_model)
train_btn.pack(pady=10)

# Input fields for amount and transaction number
amt_label = Label(root, text="Enter Transaction Amount (amt):")
amt_label.pack(pady=5)
amt_entry = Entry(root)
amt_entry.pack(pady=5)

trans_num_label = Label(root, text="Enter Transaction Number (trans_num):")
trans_num_label.pack(pady=5)
trans_num_entry = Entry(root)
trans_num_entry.pack(pady=5)

# Prediction button
predict_btn = Button(root, text="Predict Fraud", command=make_prediction)
predict_btn.pack(pady=10)

# Result labels
result_label = Label(root, text="", font=("Helvetica", 12), fg="blue")
result_label.pack(pady=10)

prediction_label = Label(root, text="", font=("Helvetica", 12))
prediction_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
