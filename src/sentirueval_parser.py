import os
import xml.etree.ElementTree as ET
import jsonpickle
from nltk.tokenize import WordPunctTokenizer
from sentence_splitter import SentenceSplitter

from src.parser import Word, PosTaggedWord, Dataset
from src.vocabulary import Vocabulary

class Aspect(object):
    def __init__(self, node):
        self.begin = int(node.get('from'))
        self.end = int(node.get('to'))
        self.target = node.get('term')
        sentiment_values = {
            'positive': 3,
            'neutral': 1,
            'negative': 0,
            'both': 2
        }
        self.polarity = sentiment_values[node.get('sentiment')]
        self.category = node.get('category')
        type_values = {
            'explicit': 0,
            'implicit': 1,
            'fct': 2
        }
        self.type = type_values[node.get('type')]
        mark_values = {
            'Rel': 0,
            'Irr': 1,
            'Cmpr': 2,
            'Prev': 3,
            'Irn': 4
        }
        self.mark = mark_values[node.get('mark')]
        self.words = []

    def __repr__(self):
        return "<Aspect {begin}:{end} {t} {category} {polarity} at {hid}>".format(
            begin=self.begin,
            end=self.end,
            category=self.category,
            polarity=self.polarity,
            t=self.type,
            hid=hex(id(self))
        )

class Review(object):
    def __init__(self, node):
        self.text = node.find(".//text").text
        self.rid = node.get("id")
        self.aspects = []
        for aspect_node in node.findall(".//aspect"):
            self.aspects.append(Aspect(aspect_node))

class SentiRuEvalDataset(Dataset):
    def parse(self, filename, grammeme_vectorizer_path):
        assert filename.endswith('xml')
        tree = ET.parse(filename)
        root = tree.getroot()
        self.reviews = []
        for review_node in root.findall(".//review"):
            self.reviews.append(Review(review_node))
        self.tokenized_reviews = self.tokenize()
        self.pos_tagged_reviews = self.pos_tag(grammeme_vectorizer_path)

    def tokenize(self):
        sentence_splitter = SentenceSplitter(language='ru')
        reviews = []
        for review in self.reviews:
            reviews.append([])
            text = review.text
            sentences = sentence_splitter.split(text)
            words_borders = list(WordPunctTokenizer().span_tokenize(text))
            for sentence in sentences:
                tokenized_sentence = []
                sentence_begin = text.find(sentence)
                sentence_end = sentence_begin + len(sentence)
                for word_begin, word_end in words_borders:
                    if word_begin >= sentence_begin and word_end <= sentence_end:
                        word_text = text[word_begin: word_end]
                        word = Word(word_text, word_begin, word_end)
                        for opinion in review.aspects:
                            if word.begin >= opinion.begin and word.end <= opinion.end:
                                word.set_opinion(opinion)
                                opinion.words.append(word)
                        tokenized_sentence.append(word)
                reviews[-1].append(tokenized_sentence)
        return reviews
