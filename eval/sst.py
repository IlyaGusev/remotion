import optparse


def read_sentiments(file_name):
    sentiments = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            sentiments.append(int(line[0]))
    return sentiments


def main():
    parser = optparse.OptionParser()
    parser.add_option('-g')
    parser.add_option('-t')
    options, args = parser.parse_args()
    gold_file_name = options.g
    test_file_name = options.t
    gold_sentiments = read_sentiments(gold_file_name)
    test_sentiments = read_sentiments(test_file_name)

    correct = 0
    count = 0
    for gold, test in zip(gold_sentiments, test_sentiments):
        count += 1
        if gold == test:
            correct += 1
    print("Accuracy: {0:.3f}".format(float(correct)/count))


if __name__ == "__main__":
    main()