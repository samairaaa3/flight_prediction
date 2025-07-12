# ✈️ Flight Price Prediction using Random Forest

## 📌 Overview

This project aims to predict flight ticket prices based on various features such as airline, source, destination, stops, duration, and more.  
It uses a **Random Forest Regressor**, fine-tuned using **RandomizedSearchCV**, and the final model is saved as a `.pkl` file for future use.

---

## 📊 Dataset

The dataset contains historical flight details, including:
- Airline  
- Source & Destination  
- Total Stops  
- Duration  
- Route  
- Additional Info  
- **Price** (target variable)

> **Note:** Data cleaning and feature engineering were performed to convert text-based features into machine-readable format.

---

## 🧠 Machine Learning Approach

- **Model Used:** Random Forest Regressor  
- **Hyperparameter Tuning:** RandomizedSearchCV  
- **Train-Test Split:** 80% training, 20% testing  
- **Evaluation Metrics:** R² Score, RMSE

---

## 📥 Download the Trained Model

Download the trained Random Forest model from the link below and place it inside the `models/` folder:

👉 [Download rf_random.pkl](https://drive.google.com/file/d/1O7Xz5N0IKYTEWdzc9aS6-IlHq-GpZ1ei/view?usp=share_link)

> **Note:** Make sure to manually download `rf_random.pkl` and place it inside your local `models/` directory before running prediction scripts.

---

## ▶️ Train the Model (Optional)
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
## load and predit
import pickle

# Load the model and make predictions
with open("models/rf_random.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)



---
### LOAD AND PREDICT
import pickle

# Load the model
with open("models/rf_random.pkl", "rb") as f:
    model = pickle.load(f)

# Predict on new data
y_pred = model.predict(X_test)

---
### SOME SCREENSHOTS SHOWING VISUALISATIONS
1. Departure Time Analysis
This bar chart visualizes how flights are distributed across different times of the day. Most flights are scheduled for early morning, followed by evening and morning slots.
<img width="1470" height="956" alt="Screenshot 2025-07-11 at 9 32 41 PM" src="https://github.com/user-attachments/assets/22b7dfc3-2f17-4ce7-80ae-84dfb0cad8b4" />


2. Price Distribution & Outliers
This set of plots displays the distribution of flight prices, including outliers. The boxplot and density plot help identify price range trends and anomalies.
<img width="1470" height="956" alt="Screenshot 2025-07-11 at 9 32 55 PM" src="https://github.com/user-attachments/assets/ac0e99fa-c149-4405-86cb-75171404ffd9" />

3. Total Stops vs Price
A scatter plot showing how the number of stops impacts flight prices. Flights with fewer stops (non-stop, 1-stop) generally tend to be more affordable.
<img width="1470" height="956" alt="Screenshot 2025-07-11 at 9 33 04 PM" src="https://github.com/user-attachments/assets/23b2235c-23ff-4754-b923-54c61d6ee7ac" />
