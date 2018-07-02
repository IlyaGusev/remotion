import pandas as pd
import matplotlib.pyplot as plt
from src.parser import Dataset, Word
from nltk.tokenize import WordPunctTokenizer
from sentence_splitter import SentenceSplitter


class Review:
    def __init__(self, rid='', text='', sentiment=0):
        self.rid = rid
        self.text = text
        self.sentences = []
        self.sentiment = sentiment


class IMDBDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.reviews = []
        self.language = "en"
        self.is_train = True

    def parse(self, filename, is_train=True):
        self.is_train = is_train
        data = pd.read_csv(filename, delimiter="\t")
        for _, row in data.iterrows():
            if 'sentiment' in row:
                self.reviews.append(Review(rid=row['id'], text=row['review'], sentiment=row['sentiment']))
            else:
                self.reviews.append(Review(rid=row['id'], text=row['review']))
        self.tokenize()

    def tokenize(self):
        sentence_splitter = SentenceSplitter(language='en')
        for i, review in enumerate(self.reviews):
            text = review.text
            sentences = sentence_splitter.split(text)
            for sentence in sentences:
                tokenized_sentence = []
                words_borders = list(WordPunctTokenizer().span_tokenize(sentence))
                for word_begin, word_end in words_borders:
                    word_text = sentence[word_begin: word_end]
                    word = Word(word_text, word_begin, word_end)
                    tokenized_sentence.append(word)
                self.reviews[i].sentences.append(tokenized_sentence)

    def print_stat(self):
        lengths = self.get_lengths()
        plt.hist(lengths, bins=1000)
        plt.title('Word count distribution for IMDB {}'.format("train" if self.is_train else "test"))
        plt.ylabel('Review count')
        plt.xlabel('Num words')
        plt.savefig("imdb_words_{}.png".format("train" if self.is_train else "test"))
        plt.clf()
        print("Num of reviews: " + str(len(self.reviews)))
        print("Max review length: " + str(max(lengths)))
