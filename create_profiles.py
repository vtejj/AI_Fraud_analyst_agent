# create_profiles.py (Version 2.0 with Time Features)

import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- CONFIGURATION ---
DATA_PATH = 'creditcard.csv'
OUTPUT_PATH = 'user_database.json'
TARGET_VARIABLE = 'Class'
N_PERSONAS = 9 # Your chosen optimal number
RANDOM_STATE = 42

def create_personas_with_time():
    """
    Creates spending personas using K-Means, now including time-of-day analysis.
    """
    print("--- Starting Persona Creation Script (v2.0 with Time Features) ---")

    # --- Load and Prepare Data ---
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    normal_df = df[df[TARGET_VARIABLE] == 0].copy()
    
    # --- FEATURE ENGINEERING: Convert Time to Hour of Day ---
    # The 'Time' feature is seconds from the first transaction. We convert it to a 24-hour clock.
    normal_df['Hour'] = (normal_df['Time'] / 3600) % 24
    normal_df['Hour'] = normal_df['Hour'].astype(int)
    
    # We will use all features except the original 'Time' and 'Class' for clustering.
    features_df = normal_df.drop(columns=[TARGET_VARIABLE, 'Time'])

    # --- Scale the Data ---
    print("Scaling features for clustering...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # --- Train K-Means Model ---
    print(f"Training final model with {N_PERSONAS} personas...")
    kmeans = KMeans(n_clusters=N_PERSONAS, random_state=RANDOM_STATE, n_init='auto')
    clusters = kmeans.fit_predict(scaled_features)
    features_df['persona_id'] = clusters

    # --- Analyze and Save the Personas ---
    print("Analyzing and saving personas...")
    personas = {}
    persona_profiles = features_df.groupby('persona_id').mean()
    
    for i, profile in persona_profiles.iterrows():
        personas[f'persona_{i}'] = profile.to_dict()
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(personas, f, indent=4)
        
    print(f"\n--- Persona Creation Complete ---")
    print(f"{N_PERSONAS} data-driven personas (with time analysis) saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    create_personas_with_time()