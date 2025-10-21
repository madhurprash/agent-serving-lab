"""
Simple script to load BrowseCompLongContext dataset and save as DataFrame in a data folder.
Reference: https://huggingface.co/datasets/openai/BrowseCompLongContext
"""

import hashlib
import base64
from pathlib import Path
from datasets import load_dataset
import pandas as pd


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def main():
    print("Loading BrowseCompLongContext dataset from HuggingFace...")

    # Load the encrypted dataset
    dataset = load_dataset("openai/BrowseCompLongContext")
    encrypted_data = dataset["train"]

    print(f"Dataset loaded with {len(encrypted_data)} rows")
    print("Decrypting dataset...")

    # Decrypt the data
    data = [
        {
            "problem": decrypt(row["problem"], row["canary"]),
            "answer": decrypt(row["answer"], row["canary"]),
            "urls": decrypt(row["urls"], row["canary"]),
        }
        for row in encrypted_data
    ]

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    print(f"\nDataset decrypted successfully!")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nDataFrame info:")
    print(df.info())

    # Create data folder if it doesn't exist
    data_folder = Path(__file__).parent / "data"
    data_folder.mkdir(exist_ok=True)

    # Save to data folder
    output_path = data_folder / "browsecomp_longcontext.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nDataFrame saved to: {output_path}")

    # Also save as CSV for easy viewing
    csv_path = data_folder / "browsecomp_longcontext.csv"
    df.to_csv(csv_path, index=False)
    print(f"DataFrame also saved as CSV to: {csv_path}")

    # Display sample of first row
    print("\n" + "="*80)
    print("Sample - First row preview:")
    print("="*80)
    print(f"\nProblem: {df['problem'][0][:200]}...")
    print(f"\nAnswer: {df['answer'][0]}")
    print(f"\nURLs (first 500 chars): {df['urls'][0][:500]}...")


if __name__ == "__main__":
    main()
