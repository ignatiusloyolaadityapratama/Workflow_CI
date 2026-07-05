import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model_ci():
    # Menggunakan DagsHub/MLflow untuk tracking
    mlflow.set_tracking_uri("https://dagshub.com/ignatiusloyolaadityapratama/Eksperimen_SML_Ignatius-loyola-aditya-pratama-siswa.mlflow")
    
    df = pd.read_csv("namadataset_preprocessing/iris_preprocessing.csv")
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        print(f"CI Engine: Memulai training. Run ID: {run.info.run_id}")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan ID agar bisa dibaca oleh GitHub Actions
        with open("RUN_ID.txt", "w") as f:
            f.write(run.info.run_id)
        
        # Wajib ada agar build-docker bisa bekerja
        mlflow.sklearn.log_model(model, "model")
        print("CI Engine: Model berhasil dilatih dan disimpan!")

if __name__ == "__main__":
    train_model_ci()