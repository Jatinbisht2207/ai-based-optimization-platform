from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data

def main():
    print("Starting AI Energy Analytics System...\n")

    df = load_data()
    df = preprocess_data(df)

    print("\nData Shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()