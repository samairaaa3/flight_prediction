# ✈️ Flight Price Prediction using Random Forest

## 📌 Overview
This project aims to predict flight ticket prices based on various features such as airline, source, destination, stops, duration, and more.  
It uses a **Random Forest Regressor**, fine-tuned using **RandomizedSearchCV**, and the final model is saved in a `.pkl` file for future use.

---

## 📊 Dataset
The dataset used contains historical flight details like:
- Airline
- Source & Destination
- Total Stops
- Duration
- Route
- Additional Info
- Price (target variable)

> **Note:** Data cleaning and feature engineering were done to convert text-based features into machine-readable format.

---

## 🧠 Machine Learning Approach
- **Algorithm:** Random Forest Regressor  
- **Tuning Method:** RandomizedSearchCV  
- **Train-Test Split:** 80% training, 20% testing  
- **Evaluation Metric:** R² Score, RMSE

---
## 🔧 How to Use

### ▶️ Train the Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Save the model
with open("models/rf_random.pkl", "wb") as f:
    pickle.dump(rf_model, f)
---

### Download Trained Model

Download the trained Random Forest model from the link below and place it inside the `models/` folder:

👉 [Download rf_random.pkl](https://drive.google.com/your-copied-link-here)
