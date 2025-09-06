import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split

print("--- Starting Agent vs. Simple Model Evaluation Script (v2.0) ---")

# Configuration 
DATA_PATH = 'creditcard.csv'
MODEL_PATH = 'fraud_model.joblib'
PERSONAS_PATH = 'user_database.json'
TARGET_VARIABLE = 'Class'
RANDOM_STATE = 42
TEST_SIZE = 0.3
ML_THRESHOLD = 0.40

# Load all necessary assets 
print("Loading assets...")
df = pd.read_csv(DATA_PATH)
pipeline = joblib.load(MODEL_PATH)
with open(PERSONAS_PATH, 'r') as f:
    PERSONAS = json.load(f)
PERSONA_CENTERS = {p_id: pd.Series(p) for p_id, p in PERSONAS.items()}

#  Recreate the exact same test set 
X = df.drop(TARGET_VARIABLE, axis=1)
y = df[TARGET_VARIABLE]
_, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_test = X_test.copy()

#  two systems we are comparing 

def get_simple_ml_decision(score):
    if score >= ML_THRESHOLD: return "FLAG"
    else: return "APPROVE"

def get_ai_agent_decision(score, transaction_data):
    """
    Simulates the decision of your intelligent agent, now with time-based analysis.
    """
    # --- NEW: We must engineer the 'Hour' feature here, just like our real tool does ---
    transaction_hour = (transaction_data['Time'] / 3600) % 24
    
    transaction_series = pd.Series(transaction_data)
    features_for_comparison_series = transaction_series.drop(['Time'])
    features_for_comparison_series['Hour'] = transaction_hour
    
    persona_features = list(PERSONA_CENTERS.values())[0].index
    transaction_features_for_comparison = features_for_comparison_series[persona_features]
    
    distances = {p_id: np.linalg.norm(transaction_features_for_comparison - c) for p_id, c in PERSONA_CENTERS.items()}
    closest_persona_id = min(distances, key=distances.get)
    persona_profile = PERSONAS[closest_persona_id]
    
    context_risk = "Low"
    persona_avg_hour = persona_profile.get('Hour', 12) # Default to midday if not found
    if (1 <= transaction_hour <= 5) and (7 <= persona_avg_hour <= 22):
        context_risk = "High"
    elif transaction_series['Amount'] > (persona_profile['Amount'] * 10):
        context_risk = "High"
    elif transaction_series['Amount'] < 2.00:
        context_risk = "Medium"

    
    if score >= 0.80: return "BLOCK"
    if score >= ML_THRESHOLD and context_risk == "High": return "BLOCK"
    if score >= ML_THRESHOLD: return "CHALLENGE"
    return "APPROVE"

# Main Evaluation Logic
print("Evaluating model on the test set...")
X_test['fraud_score'] = pipeline.predict_proba(X_test)[:, 1]
X_test['true_class'] = y_test

grey_area_df = X_test[(X_test['fraud_score'] >= ML_THRESHOLD) & (X_test['fraud_score'] < 0.80)].copy()
print(f"\nFound {len(grey_area_df)} transactions in the 'grey area' (score between {ML_THRESHOLD} and 0.80).")

grey_area_df['simple_ml_decision'] = grey_area_df['fraud_score'].apply(get_simple_ml_decision)
grey_area_df['ai_agent_decision'] = grey_area_df.apply(lambda row: get_ai_agent_decision(row['fraud_score'], row.drop(['fraud_score', 'true_class']).to_dict()), axis=1)

# Calculate the Improvement 
simple_ml_flags = grey_area_df[grey_area_df['simple_ml_decision'] == 'FLAG']
agent_blocks_fraud = grey_area_df[(grey_area_df['ai_agent_decision'] == 'BLOCK') & (grey_area_df['true_class'] == 1)]

improvement_count = len(agent_blocks_fraud)
total_grey_area_frauds = len(simple_ml_flags[simple_ml_flags['true_class'] == 1])

if total_grey_area_frauds > 0:
    improvement_percentage = (improvement_count / total_grey_area_frauds) * 100
else:
    improvement_percentage = 0

print("\n--- PERFORMANCE IMPROVEMENT ANALYSIS ---")
print(f"Total Frauds in Grey Area: {total_grey_area_frauds}")
print(f"Frauds Correctly Escalated to BLOCK by AI Agent: {improvement_count}")
print(f"Improvement Rate: The AI Agent correctly escalated {improvement_percentage:.2f}% of high-ambiguity fraudulent transactions to an immediate BLOCK.")
print("\nThis number is your proof of the value added by the contextual analysis.")