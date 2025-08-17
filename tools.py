# tools.py (Version 5.0 - The Final Version)

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


# --- TOOLS THAT PROVIDE RAW ANALYSIS ---

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
    """Provides a contextual risk analysis based on user personas."""
    try:
        transaction_series = pd.Series(transaction_data)
        persona_features = list(PERSONA_CENTERS.values())[0].index
        transaction_features_for_comparison = transaction_series[persona_features]
        distances = {
            p_id: np.linalg.norm(transaction_features_for_comparison - center)
            for p_id, center in PERSONA_CENTERS.items()
        }
        closest_persona_id = min(distances, key=distances.get)
        closest_persona_profile = PERSONAS[closest_persona_id]
        persona_avg_amount = closest_persona_profile['Amount']
        transaction_amount = transaction_series['Amount']

        # The corrected version of that block...

        if transaction_amount > (persona_avg_amount * 10):
            return {
                "context_risk": "High", 
                "reason": f"Amount ${transaction_amount:,.2f} is 10x higher than average for its persona ('{closest_persona_id}').",
                "persona_name": closest_persona_id  # <-- ADD THIS LINE
            }
        if transaction_amount < 2.00:
             return {
                "context_risk": "Medium", 
                "reason": "Amount is suspiciously small (potential card testing).",
                "persona_name": closest_persona_id  # <-- ADD THIS LINE
             }
        return {
            "context_risk": "Low", 
            "reason": f"Transaction is consistent with its persona ('{closest_persona_id}').",
            "persona_name": closest_persona_id  # <-- ADD THIS LINE
        }
    except Exception as e:
        return {"error": f"Context analysis failed: {str(e)}"}