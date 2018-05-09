import os
import xml.etree.ElementTree as ET
import jsonpickle
from nltk.tokenize import WordPunctTokenizer
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer

from src.parser import Word, PosTaggedWord
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

class Review(object):
    def __init__(self, node):
        self.rid = node.get('rid')
        self.sentences = []
        for sentence_node in node.findall(".//sentence"):
            self.sentences.append(Sentence(sentence_node))

class Dataset(object):
    def __init__(self, filename, language, grammeme_vectorizer_path=None):
        self.language = language
        if filename.endswith('xml'):
            tree = ET.parse(filename)
            root = tree.getroot()
            self.reviews = []
            for review_node in root.findall(".//Review"):
                self.reviews.append(Review(review_node))
            self.tokenized_reviews = self.__tokenize()
            self.pos_tagged_reviews = self.__pos_tag(grammeme_vectorizer_path)
        elif filename.endswith('json'):
            with open(filename, "r", encoding='utf-8') as f:
                dataset = jsonpickle.decode(f.read())
                self.__dict__.update(dataset.__dict__)

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as w:
            w.write(jsonpickle.encode(self))

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

    def __pos_tag(self, grammeme_vectorizer_path):
        if self.language == 'ru' and grammeme_vectorizer_path is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            predictor = RNNMorphPredictor()
            grammeme_vectorizer = GrammemeVectorizer(grammeme_vectorizer_path)
            pos_tagged_reviews = []
            for review in self.tokenized_reviews:
                pos_tagged_reviews.append([])
                for sentence in review:
                    words = [word.text for word in sentence]
                    forms = predictor.predict_sentence_tags(words)
                    pos_tagged_sentence = []
                    for word_idx, form in enumerate(forms):
                        vector = grammeme_vectorizer.get_vector(form.pos + "#" + form.tag)
                        pos_tagged_sentence.append(PosTaggedWord(sentence[word_idx], form.pos, form.tag, vector))
                    pos_tagged_reviews[-1].append(pos_tagged_sentence)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return pos_tagged_reviews
        raise NotImplementedError()

    def get_vocabulary(self):
        vocabulary = Vocabulary()
        for review in self.tokenized_reviews:
            for sentence in review:
                vocabulary.add_sentence(" ".join([word.text for word in sentence]))
        return vocabulary


