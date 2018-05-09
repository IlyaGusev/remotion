import xml.etree.ElementTree as ET
from nltk.tokenize import WordPunctTokenizer

from vocabulary import Vocabulary

class Opinion:
    def __init__(self, node):
        self.begin = int(node.get('from'))
        self.end = int(node.get('to'))
        self.target = node.get('target')
        polarity = {
            'positive': 3,
            'neutral': 1,
            'negative': 0,
            'conflict': 2
        }
        self.polarity = polarity[node.get('polarity')]
        category = node.get('category')
        self.cat_first = category.split('#')[0]
        self.cat_second = category.split('#')[1]

    def __repr__(self):
        return "<Opinion {begin}:{end} {c1}#{c2} {polarity} at {hid}>".format(
            begin=self.begin,
            end=self.end,
            c1=self.cat_first,
            c2=self.cat_second,
            polarity=self.polarity,
            hid=hex(id(self))
        )

class Sentence:
    def __init__(self, node):
        self.text = node.find(".//text").text
        self.opinions = []
        for opinion_node in node.findall(".//Opinion"):
            self.opinions.append(Opinion(opinion_node))

class Review:
    def __init__(self, node):
        self.rid = node.get('rid')
        self.sentences = []
        for sentence_node in node.findall(".//sentence"):
            self.sentences.append(Sentence(sentence_node))

class Word:
    def __init__(self, text, begin, end):
        self.text = text
        self.begin = begin
        self.end = end
        self.opinion = None

    def set_opinion(self, opinion):
        self.opinion = opinion

    def get_polarity(self):
        if self.opinion is None:
            return 0
        else:
            self.opinion.polarity

    def is_colored(self):
        return self.opinion is not None

    def __repr__(self):
        return '<Word "{text}" from {begin} to {end} with opinion {opinion} at {hid}>'.format(
            text = self.text,
            begin = self.begin,
            end = self.end,
            opinion = self.opinion,
            hid = hex(id(self))
        )

class Dataset:
    def __init__(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        self.reviews = []
        for review_node in root.findall(".//Review"):
            self.reviews.append(Review(review_node))
        self.tokenized_reviews = self.__tokenize()

    def get_opinion_count(self):
        count = 0
        for review in self.reviews:
            for sentence in review.sentences:
                for opinion in sentence.opinions:
                    count += 1
        return count

    def __tokenize(self):
        reviews = []
        current_rid = None
        for review in self.reviews:
            reviews.append([])
            for sentence in review.sentences:
                text = sentence.text
                words_borders = list(WordPunctTokenizer().span_tokenize(text))
                tokenized_sentence = []
                for word_begin, word_end in words_borders:
                    word_text = text[word_begin: word_end]
                    word = Word(word_text, word_begin, word_end)
                    for opinion in sentence.opinions:
                        if opinion.target == 'NULL' or opinion.begin == 0 and opinion.end == 0:
                            continue
                        if word.begin >= opinion.begin and word.end <= opinion.end:
                            word.set_opinion(opinion)
                    tokenized_sentence.append(word)
                reviews[-1].append(tokenized_sentence)
        return reviews

    def get_vocabulary(self):
        vocabulary = Vocabulary()
        for review in self.tokenized_reviews:
            for sentence in review:
                vocabulary.add_sentence(" ".join([word.text for word in sentence]))
        return vocabulary
