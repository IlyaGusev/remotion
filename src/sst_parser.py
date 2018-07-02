import matplotlib.pyplot as plt
from src.parser import Dataset, Word
from nltk.tokenize import WordPunctTokenizer


class Review:
    def __init__(self, text='', sentiment=0):
        self.text = text
        self.sentiment = sentiment
        self.sentences = []


class SSTDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.reviews = []
        self.language = "en"
        self.is_train = True

    def parse(self, filename, is_train=True):
        self.is_train = is_train
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                sentiment = int(line.split(" ")[0])
                text = " ".join(line.split(" ")[1:])
                self.reviews.append(Review(text=text, sentiment=sentiment))
        self.tokenize()
        self.pos_tag()

    def tokenize(self):
        for i, review in enumerate(self.reviews):
            text = review.text
            words_borders = list(WordPunctTokenizer().span_tokenize(text))
            tokenized_sentence = []
            for word_begin, word_end in words_borders:
                word_text = text[word_begin: word_end]
                word = Word(word_text, word_begin, word_end)
                tokenized_sentence.append(word)
            review.sentences.append(tokenized_sentence)

    def print_stat(self):
        lengths = self.get_lengths()
        plt.hist(lengths, bins=1000)
        plt.title('Word count distribution for SST {}'.format("train" if self.is_train else "test"))
        plt.ylabel('Review count')
        plt.xlabel('Num words')
        plt.savefig("imdb_words_{}.png".format("train" if self.is_train else "test"))
        plt.clf()
        print("Num of reviews: " + str(len(self.reviews)))
        print("Max review length: " + str(max(lengths)))
