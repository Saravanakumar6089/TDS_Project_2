import chardet
import pandas as pd
# Detect the file encoding
filepath="happiness/happiness.csv"
with open(filepath, 'rb') as file:
    result = chardet.detect(file.read())  # Read a portion of the file
    detected_encoding = result['encoding']
    print(f"Detected encoding: {detected_encoding}")

# Use the detected encoding to load the file
try:
    df = pd.read_csv(filepath, encoding=detected_encoding)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
