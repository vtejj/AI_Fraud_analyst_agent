# Imports
print("Importing libraries for XGBoost model...")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  
from imblearn.over_sampling import SMOTE
print("Imports complete.")

# Configuration
DATA_PATH = 'creditcard.csv'
MODEL_PATH = 'fraud_model_xgb.joblib' 
TARGET_VARIABLE = 'Class'
TEST_SIZE = 0.3
RANDOM_STATE = 42

def main():
    """Main function to train and save the XGBoost model pipeline."""
    print("--- Starting XGBoost Training Script ---")
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preparing data...")
    X = df.drop(TARGET_VARIABLE, axis=1)
    y = df[TARGET_VARIABLE]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Data split complete.")

    #  DEFINE AND TRAIN XGBOOST MODEL
    print("Defining XGBoost model pipeline...")
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'))
    ])

    print("Training the XGBoost pipeline...")
    pipeline.fit(X_train, y_train)
    
    # 6. SAVE THE PIPELINE 
    print(f"Saving final XGBoost pipeline to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    
    print("--- XGBoost Script Finished Successfully ---")
    print(f"Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    main()