import os
import xml.etree.ElementTree as ET
import jsonpickle
from nltk.tokenize import WordPunctTokenizer
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer

from src.parser import Word, PosTaggedWord, Dataset
from src.vocabulary import Vocabulary

class Opinion(object):
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
        self.words = []

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
        self.aspects = []
        for opinion_node in node.findall(".//Opinion"):
            self.aspects.append(Opinion(opinion_node))

class Review(object):
    def __init__(self, node):
        self.rid = node.get('rid')
        self.sentences = []
        self.aspects = []
        for sentence_node in node.findall(".//sentence"):
            self.sentences.append(Sentence(sentence_node))
            self.aspects += self.sentences[-1].aspects

class SemEvalDataset(Dataset):
    def parse(self, filename, grammeme_vectorizer_path):
        assert filename.endswith('xml')
        tree = ET.parse(filename)
        root = tree.getroot()
        self.reviews = []
        for review_node in root.findall(".//Review"):
            self.reviews.append(Review(review_node))
        self.tokenized_reviews = self.tokenize()
        self.pos_tagged_reviews = self.pos_tag(grammeme_vectorizer_path)

    def tokenize(self):
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
                    for opinion in sentence.aspects:
                        if opinion.target == 'NULL' or opinion.begin == 0 and opinion.end == 0:
                            continue
                        if word.begin >= opinion.begin and word.end <= opinion.end:
                            word.set_opinion(opinion)
                            self.words.append(word)
                    tokenized_sentence.append(word)
                reviews[-1].append(tokenized_sentence)
        return reviews

