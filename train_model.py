from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 2. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 3. Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
rf_model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 7. Save the trained model
joblib.dump(rf_model, "random_forest_iris_model.pkl")

print("Model saved as random_forest_iris_model.pkl")
