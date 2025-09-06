import pandas as pd
import json

# Configuration 
DATA_PATH = 'creditcard.csv'
OUTPUT_PATH = 'genuine_samples.txt'
TARGET_VARIABLE = 'Class'
NUM_SAMPLES = 10

def generate_genuine_samples():
    """
    Reads your specific CSV file, finds 10 GENUINE (non-fraudulent) transactions,
    and saves them to a text file for easy copy-pasting.
    """
    try:
        print(f"--- Generating 10 Genuine Samples from '{DATA_PATH}' ---")
        df = pd.read_csv(DATA_PATH)

        genuine_df = df[df[TARGET_VARIABLE] == 0].copy()

        if genuine_df.empty:
            print("No genuine transactions found in the dataset.")
            return

        genuine_samples = genuine_df.sample(n=NUM_SAMPLES, random_state=202) # Use a new random state

        with open(OUTPUT_PATH, 'w') as f:
            f.write("--- 10 JSON Objects for Genuine Transactions (from your file) ---\n\n")
            
            for i, (index, row) in enumerate(genuine_samples.iterrows()):
                # Drop the 'Class' column as the API doesn't expect it
                row_dict = row.drop(TARGET_VARIABLE).to_dict()
                
                # Convert to a double-quoted JSON string for the API
                json_output = json.dumps(row_dict)
                
                f.write(f"--- Transaction Sample {i + 1} (Row {index} from your CSV) ---\n")
                f.write(json_output)
                f.write("\n" + "-" * 50 + "\n\n")

        print(f"--- SUCCESS! ---")
        print(f"Your 10 unique, verifiable genuine samples have been saved to '{OUTPUT_PATH}'.")

    except FileNotFoundError:
        print(f"Error: The file '{DATA_PATH}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_genuine_samples()