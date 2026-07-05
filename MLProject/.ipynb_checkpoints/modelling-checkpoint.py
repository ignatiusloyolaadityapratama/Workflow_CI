import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    mlflow.set_tracking_uri("https://dagshub.com/ignatiusloyolaadityapratama/Eksperimen_SML_Ignatius-loyola-aditya-pratama-siswa.mlflow")
    
    # Path dataset disesuaikan (coba cari file csv-nya di folder yang sama)
    df = pd.read_csv("namadataset_preprocessing/iris_preprocessing.csv")
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan ID
        with open("RUN_ID.txt", "w") as f:
            f.write(run.info.run_id)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        print("Training Selesai. Run ID:", run.info.run_id)

if __name__ == "__main__":
    train_model()