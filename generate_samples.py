# generate_samples.py

import pandas as pd
import json

# --- Configuration ---
DATA_PATH = 'creditcard.csv'
OUTPUT_PATH = 'fraud_samples.txt'
TARGET_VARIABLE = 'Class'
NUM_SAMPLES = 10

def generate_fraudulent_samples():
    """
    Reads your specific CSV file, finds 10 fraudulent transactions,
    and saves them to a text file for easy copy-pasting.
    """
    try:
        print(f"--- Generating 10 Fraudulent Samples from '{DATA_PATH}' ---")
        df = pd.read_csv(DATA_PATH)

        fraudulent_df = df[df[TARGET_VARIABLE] == 1].copy()

        if len(fraudulent_df) < NUM_SAMPLES:
            print(f"Warning: Found only {len(fraudulent_df)} fraudulent transactions.")
            samples_to_take = len(fraudulent_df)
        else:
            samples_to_take = NUM_SAMPLES

        fraud_samples = fraudulent_df.sample(n=samples_to_take, random_state=101) # Use a new random state

        with open(OUTPUT_PATH, 'w') as f:
            f.write("--- 10 JSON Objects for Fraudulent Transactions (from your file) ---\n\n")

            for i, (index, row) in enumerate(fraud_samples.iterrows()):
                row_dict = row.drop(TARGET_VARIABLE).to_dict()
                # NEW, CORRECT LINE
                # NEW, CORRECT LINE
                json_output = json.dumps(row_dict)

                f.write(f"--- Transaction Sample {i + 1} (Row {index} from your CSV) ---\n")
                f.write(json_output)
                f.write("\n" + "-" * 50 + "\n\n")

        print(f"--- SUCCESS! ---")
        print(f"Your 10 unique, verifiable fraud samples have been saved to '{OUTPUT_PATH}'.")

    except FileNotFoundError:
        print(f"Error: The file '{DATA_PATH}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_fraudulent_samples()