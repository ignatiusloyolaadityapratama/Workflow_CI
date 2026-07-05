import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model_ci():
    mlflow.autolog(log_input_examples=True, log_model_signatures=True)

    # Membaca dataset dari folder lokal MLProject
    df = pd.read_csv("namadataset_preprocessing/iris_preprocessing.csv")
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        print("CI Engine: Memulai training model otomatis...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("CI Engine: Model berhasil dilatih dan direkam ke MLflow DagsHub!")

if __name__ == "__main__":
    train_model_ci()