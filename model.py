print("Importing libraries...")
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report

print("Imports complete.")

# CONFIGURATION 
DATA_PATH = 'creditcard.csv'
MODEL_PATH = 'fraud_model.joblib'
TARGET_VARIABLE = 'Class'
TEST_SIZE = 0.3
RANDOM_STATE = 42
BUSINESS_THRESHOLD = 0.40

def main():
    """Main function to train, evaluate, and save the model pipeline."""
    print("--- Starting Training & Evaluation Script ---")
    
    #  3. LOAD & PREPARE DATA 
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preparing data...")
    X = df.drop(TARGET_VARIABLE, axis=1)
    y = df[TARGET_VARIABLE]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Data split into training and testing sets.")

    # 4. DEFINE AND TRAIN MODEL 
    print("Defining model pipeline...")
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))
    ])

    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    # 5. SAVE THE PIPELINE
    print(f"Saving final model pipeline to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

    # 6. COMPREHENSIVE MODEL EVALUATION
    print("\n--- Starting Comprehensive Model Evaluation on Test Set ---")
    
    test_set_size = len(X_test)
    print(f"Test Set Size: {test_set_size} transactions")

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC Score: {pr_auc:.4f}")

    print(f"\n--- Performance at Business Threshold ({BUSINESS_THRESHOLD}) ---")
    y_pred_thresholded = (y_pred_proba >= BUSINESS_THRESHOLD).astype(int)
    
    report = classification_report(y_test, y_pred_thresholded, target_names=['Genuine (0)', 'Fraud (1)'])
    print(report)
    
    print("--- Script Finished Successfully ---")

if __name__ == "__main__":
    main()