# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("diabetes.csv")

# Split inputs (X) and output (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save trained model to file
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved!")
