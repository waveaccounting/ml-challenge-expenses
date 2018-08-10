from data_preparation import Data
if __name__ == "__main__":
    training_data_example = "./training_data_example.csv"
    train_data = Data(training_data_example)
    training_features = train_data.create_features()
    training_targets = train_data.create_targets()
    print(training_features)
