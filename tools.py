# tools.py (Version 6.1 with Time-Based Rules)

import joblib
import pandas as pd
from langchain.tools import tool
import json
import numpy as np

# --- Load Models and Profiles at Startup ---
MODEL = joblib.load('fraud_model.joblib')
with open('user_database.json', 'r') as f:
    PERSONAS = json.load(f)
PERSONA_CENTERS = {p_id: pd.Series(p) for p_id, p in PERSONAS.items()}

# --- AGENT TOOLS ---

@tool
def get_ml_model_analysis(transaction_data: dict) -> dict:
    """Provides a fraud probability score from the ML model."""
    try:
        input_data = pd.DataFrame([transaction_data])
        probability_score = MODEL.predict_proba(input_data)[0][1]
        return {"fraud_probability_score": float(probability_score)}
    except Exception as e:
        return {"error": f"ML model failed: {str(e)}"}

@tool
def get_contextual_analysis(transaction_data: dict) -> dict:
    """Provides a contextual risk analysis based on user personas and time of day."""
    try:
        # --- NEW: Time-based analysis ---
        transaction_hour = (transaction_data['Time'] / 3600) % 24
        
        transaction_series = pd.Series(transaction_data)
        
        # Add the 'Hour' to the series for persona matching
        features_for_comparison_series = transaction_series.drop(['Time'])
        features_for_comparison_series['Hour'] = transaction_hour

        persona_features = list(PERSONA_CENTERS.values())[0].index
        transaction_features_for_comparison = features_for_comparison_series[persona_features]

        distances = {p_id: np.linalg.norm(transaction_features_for_comparison - c) for p_id, c in PERSONA_CENTERS.items()}
        closest_persona_id = min(distances, key=distances.get)
        
        persona_profile = PERSONAS[closest_persona_id]
        persona_avg_hour = persona_profile['Hour']
        transaction_amount = transaction_series['Amount']
        persona_avg_amount = persona_profile['Amount']

        # --- NEW TIME-BASED RULE ---
        # If transaction is during late night hours (1am-5am) AND the persona's avg hour is during the day
        if (1 <= transaction_hour <= 5) and (7 <= persona_avg_hour <= 22):
            return {
                "context_risk": "High",
                "reason": f"Transaction occurred at a suspicious time ({int(transaction_hour)}:00) for its user persona ('{closest_persona_id}')."
            }

        if transaction_amount > (persona_avg_amount * 10):
            return {"context_risk": "High", "reason": f"Amount ${transaction_amount:,.2f} is 10x higher than average for its persona ('{closest_persona_id}')."}
        if transaction_amount < 2.00:
             return {"context_risk": "Medium", "reason": "Amount is suspiciously small (potential card testing)."}
        
        return {"context_risk": "Low", "reason": f"Transaction is consistent with its persona ('{closest_persona_id}')."}
    except Exception as e:
        return {"error": f"Context analysis failed: {str(e)}"}