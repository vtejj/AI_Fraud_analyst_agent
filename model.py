#model.py

print("Importing libraries...")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline # Use imblearn's Pipeline
print("Imports complete.")

DATA_PATH = 'creditcard.csv'
MODEL_PATH = 'fraud_model.joblib'
TARGET_VARIABLE = 'Class'

def main():
    print("--- Starting Training Script ---")
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preparing data...")
    X = df.drop(TARGET_VARIABLE, axis=1)
    y = df[TARGET_VARIABLE]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")

    # --- 5. DEFINE AND TRAIN MODEL (Your best model: SMOTE + RandomForest) ---
    # We will use a Pipeline to combine SMOTE, scaling, and the classifier.
    # This is best practice and ensures the same steps are applied during prediction.
    print("Defining model pipeline...")
    
    # The pipeline applies SMOTE only to the training data during the .fit() step,
    # which is the correct way to prevent data leakage.
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # --- 6. SAVE THE PIPELINE ---
    # We save the entire pipeline. It "remembers" all the steps.
    print(f"Saving final model pipeline to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    
    print("--- Script Finished Successfully ---")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()