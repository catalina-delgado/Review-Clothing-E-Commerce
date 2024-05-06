# Mariline Catalina Delgado Martínez
# Analítica de Datos
# Mayo 2024
#%% carga de datos
import pandas as pd

db = pd.read_csv('content/validation/Womens Clothing E-Commerce Reviews.csv')
db.info()

#%% preprocesamiento

db=db[['Clothing ID', 'Age', 'Review Text', 'Class Name', 'Positive Feedback Count']]
db['Clothing ID']=db['Clothing ID'].astype(str)
db = db.dropna(subset=['Class Name'])
db = db.dropna(subset=['Positive Feedback Count'])

X = db[['Age','Positive Feedback Count']]
y = db[['Class Name']]
#%% mlflow
import mlflow
from mlflow.models import infer_signature

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

#%% modelo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs", #solucionador
    "max_iter": 1000, #numero maximo de iteraciones
    "multi_class": "auto", #metodo de clasificacion
    "random_state": 8888, #semilla aleatoria
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

#%% Start an MLflow run

with mlflow.start_run():

    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="PositiveScore",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    Class_feature_names = db['Class Name']

    result = pd.DataFrame(X_test, columns=Class_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    result[:4]

# %%
