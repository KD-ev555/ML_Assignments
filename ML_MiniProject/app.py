from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import logging

# Initialize the Flask app
app = Flask(__name__)

# Set up global variables
trained_model = None
scaler = None

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to load the model and scaler if they exist
def load_model():
    global trained_model, scaler
    try:
        trained_model = joblib.load('trained_model.pkl')
        scaler = joblib.load('scaler.pkl')
        logging.info("Model and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")

# Preprocess the data
def preprocess_data(df):
    global scaler
    # Remove unnamed columns if present
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
    label_encoders = {}

    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Ensure the 'is_fraud' column is present
    if 'is_fraud' not in df.columns:
        raise ValueError("'is_fraud' column missing in dataset")

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Train the model based on user selection
@app.route('/train', methods=['POST'])
def train_model():
    global trained_model, scaler
    try:
        # Get the model choice from the form
        model_choice = request.form['model_choice']

        # Load the predefined dataset
        df = pd.read_csv('Book2.csv')

        # Preprocess the data
        X, y = preprocess_data(df)

        # Train the selected model
        if model_choice == "LogisticRegression":
            trained_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif model_choice == "DecisionTree":
            trained_model = DecisionTreeClassifier(random_state=42)
        elif model_choice == "RandomForest":
            trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "KNN":
            trained_model = KNeighborsClassifier(n_neighbors=5)

        trained_model.fit(X, y)

        # Save model and scaler
        joblib.dump(trained_model, 'trained_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')

        # Reload the model
        load_model()

        return jsonify({'result': 'Model trained successfully!'})

    except Exception as e:
        logging.error(f"Error training model: {e}")
        return jsonify({'error': str(e)})

# Make predictions with the trained model
@app.route('/predict', methods=['POST'])
def predict():
    global trained_model, scaler
    try:
        # Validate input values
        amt = request.form.get('amt')
        trans_num = request.form.get('trans_num')

        if amt is None or trans_num is None:
            return jsonify({'error': 'Missing input data.'})

        try:
            amt = float(amt)
            trans_num = int(trans_num)
        except ValueError:
            return jsonify({'error': 'Invalid input format.'})

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[amt, trans_num]], columns=['amt', 'trans_num'])

        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = trained_model.predict(input_scaled)

        # Determine prediction result
        if prediction[0] == 1:
            return jsonify({'result': 'Fraud detected (Yes)', 'color': 'red'})
        else:
            return jsonify({'result': 'No fraud detected (No)', 'color': 'green'})

    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)})

# Render the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    load_model()  # Load model and scaler when the server starts
    app.run(debug=True)
