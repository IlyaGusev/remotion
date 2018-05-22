import sys
from src.config import Config, get_targets_additionals
from src.parser import get_data
from src.train import train_model
from src.predict import predict

def main(config_filename):
    config = Config()
    config.load(config_filename)

    train_data, test_data = get_data(config.data_config)
    max_length = max(train_data.get_lengths() + test_data.get_lengths())
    vocabulary = train_data.get_vocabulary().merge(test_data.get_vocabulary())
    char_set = train_data.get_char_set()
    print(vocabulary.size())
    print(char_set)

    targets, additionals, rev_categories = get_targets_additionals(train_data)
    train_model(config_filename, train_data, vocabulary, char_set, targets, additionals)
    predict(config_filename, test_data, vocabulary, char_set,
            targets, additionals, rev_categories)

if __name__ == "__main__":
    config_filename = sys.argv[1]
    main(config_filename)
