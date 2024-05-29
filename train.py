
def train():
    print('Training model...')

def main():
    # Load data
    data = pd.read_csv('data.csv')
    # Preprocess data
    data = preprocess(data)
    # Train model
    model = train(data)
    # Save model
    model.save('model.pkl')

if __name__ == "__main__":
    main()
