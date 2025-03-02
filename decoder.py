import pandas as pd
import pickle

# === Load Encoders ===
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# === Function to Decode Encoded Values ===
def decode_column(column, encoder):
    """Convert encoded values back to their original text labels."""
    return encoder.inverse_transform(column.astype(int))  # Ensure it's integer

# === Decode and Save Both Anomalies and Attacks ===
for filename in ["detected_anomalies.csv", "detected_attacks.csv"]:
    try:
        # Load file
        data = pd.read_csv(filename)

        # Decode categorical columns
        for col in ["protocol_type", "service", "flag"]:
            if col in data.columns:
                data[col] = decode_column(data[col], encoders[col])

        # Save decoded data
        output_filename = filename.replace(".csv", "_text.csv")
        data.to_csv(output_filename, index=False)
        print(f"✅ Decoded file saved as `{output_filename}`")

    except FileNotFoundError:
        print(f"⚠️ Warning: `{filename}` not found. Run `IDS.py` first to generate results.")
