# create_profiles.py

import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

# Suppress a warning from KMeans about memory leaks on Windows
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- CONFIGURATION ---
DATA_PATH = 'creditcard.csv'
OUTPUT_PATH = 'user_database.json'
TARGET_VARIABLE = 'Class'
MAX_PERSONAS_TO_TEST = 20 # We'll test up to 20 clusters to find the best number

def create_personas():
    """
    Analyzes transaction data to create data-driven user personas using K-Means clustering
    and the Elbow Method to find the optimal number of personas.
    """
    print("--- Starting Persona Creation Script ---")

    # --- Load and Prepare Data ---
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # We only create personas from NON-FRAUDULENT data for a clean baseline.
    normal_df = df[df[TARGET_VARIABLE] == 0].copy()
    
    # We don't need 'Time' or 'Class' for clustering behavior.
    features_df = normal_df.drop(columns=[TARGET_VARIABLE, 'Time'])

    # --- Scale the Data (Crucial for K-Means) ---
    print("Scaling features for clustering...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # --- Find the Optimal Number of Personas (The Elbow Method) ---
    print(f"Finding the optimal number of personas (testing up to {MAX_PERSONAS_TO_TEST})...")
    inertias = []
    k_range = range(1, MAX_PERSONAS_TO_TEST + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        print(f"  Tested {k} personas...")

    # Plot the Elbow Method graph to visualize the best 'k'
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertias, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Personas')
    plt.xlabel('Number of Personas (k)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig('elbow_plot.png')
    print("\nAn 'elbow_plot.png' has been saved to your project folder.")
    print("Please open it to find the 'elbow' point where the line starts to flatten.")
    
    # Ask the user for the best 'k' after they've seen the plot
    optimal_k = int(input(">>> Please enter the optimal number of personas (k) based on the plot: "))

    # --- Train the Final K-Means Model ---
    print(f"\nTraining final model with {optimal_k} personas...")
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    clusters = final_kmeans.fit_predict(scaled_features)

    features_df['persona_id'] = clusters

    # --- Analyze and Save the Personas ---
    print("Analyzing and saving personas...")
    personas = {}
    # We will save the "average" profile for each persona
    persona_profiles = features_df.groupby('persona_id').mean()
    
    for i, profile in persona_profiles.iterrows():
        personas[f'persona_{i}'] = profile.to_dict()
    
    # Save the final personas dictionary to our JSON database
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(personas, f, indent=4)
        
    print(f"\n--- Persona Creation Complete ---")
    print(f"{optimal_k} data-driven personas have been saved to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    create_personas()