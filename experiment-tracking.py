import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import os
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Setup MLflow Connection
# Ensure your server is running: mlflow server --host 127.0.0.1 --port 5000
mlflow.set_tracking_uri(uri='http://localhost:5000')
mlflow.set_experiment('MLFlow Quickstart')

# 2. Prepare Data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define Parameters & Train Model
# Removed 'multi_class' to avoid TypeError in newer sklearn versions
params = {
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': 8888
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# 4. Evaluate Model
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 5. Create Confusion Matrix Artifact
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay.from_estimator(
    lr, X_test, y_test, 
    display_labels=iris.target_names, 
    cmap=plt.cm.Blues, 
    ax=ax
)
plt.title("Confusion Matrix - Iris Model")
plot_path = "confusion_matrix.png"
plt.savefig(plot_path)
plt.close() # Close plot to save memory

# 6. Log to MLflow
with mlflow.start_run():
    # Log hyperparameters and metrics
    mlflow.log_params(params)
    mlflow.log_metric('accuracy', accuracy)

    # Log the visual plot artifact
    mlflow.log_artifact(plot_path)

    # Create a signature (schema) for the model
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model and register it
    # Note: 'name' is used instead of 'artifact_path' to avoid deprecation warnings
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name='tracking_quickstart'
    )

    # Add descriptive tags
    mlflow.set_tag('training_info', 'Logistic Regression with Confusion Matrix')

# 7. Cleanup local plot file
if os.path.exists(plot_path):
    os.remove(plot_path)

# 8. Loading the model back to verify
print(f"Loading model from: {model_info.model_uri}")
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

# Display Sample Results
result = pd.DataFrame(X_test, columns=iris.feature_names)
result['actual_class'] = y_test
result['predicted_class'] = predictions

print("\n--- Sample Predictions ---")
print(result.head(3))
print(f"\nModel successfully registered as Version {model_info.registered_model_version}")