import os
import xml.etree.ElementTree as ET
import jsonpickle
from nltk.tokenize import WordPunctTokenizer
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer

from src.parser import Word, PosTaggedWord
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
        self.aspects = []
        for aspect_node in node.findall(".//aspect"):
            self.aspects.append(Aspect(aspect_node))

class Dataset(object):
    def __init__(self, filename, language, grammeme_vectorizer_path=None):
        self.language = language
        if filename.endswith('xml'):
            tree = ET.parse(filename)
            root = tree.getroot()
            self.reviews = []
            for review_node in root.findall(".//review"):
                self.reviews.append(Review(review_node))
            self.tokenized_reviews = self.__tokenize()
        elif filename.endswith('json'):
            with open(filename, "r", encoding='utf-8') as f:
                dataset = jsonpickle.decode(f.read())
                self.__dict__.update(dataset.__dict__)

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as w:
            w.write(jsonpickle.encode(self))

    def get_opinion_count(self):
        return sum([len(review.aspects) for review in self.reviews])

    def __tokenize(self):
        reviews = []
        for review in self.reviews:
            reviews.append([])
            text = review.text
            words_borders = list(WordPunctTokenizer().span_tokenize(text))
            for word_begin, word_end in words_borders:
                word_text = text[word_begin: word_end]
                word = Word(word_text, word_begin, word_end)
                for opinion in review.aspects:
                    if opinion.begin == 0 and opinion.end == 0:
                        continue
                    if word.begin >= opinion.begin and word.end <= opinion.end:
                        word.set_opinion(opinion)
                reviews[-1].append(word)
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
            vocabulary.add_sentence(" ".join([word.text for word in review]))
        return vocabulary


