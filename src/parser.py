import os
import jsonpickle
from rnnmorph.predictor import RNNMorphPredictor
import spacy

from src.vocabulary import Vocabulary


class Word(object):
    def __init__(self, text, begin, end):
        self.text = text
        self.begin = begin
        self.end = end
        self.opinions = []
        self.sid = ""

    def add_opinion(self, opinion):
        self.opinions.append(opinion)

    def set_sid(self, sid):
        self.sid = sid

    def __repr__(self):
        return '<Word "{text}" from {begin} to {end} with opinions {opinions} at {hid}>'.format(
            text=self.text,
            begin=self.begin,
            end=self.end,
            opinions=self.opinions,
            hid=hex(id(self))
        )


class PosTaggedWord(Word):
    def __init__(self, word, pos, tag, vector):
        Word.__init__(self, word.text, word.begin, word.end)
        self.opinions = word.opinions
        self.sid = word.sid
        self.pos = pos
        self.tag = tag
        self.vector = vector

    def __repr__(self):
        return '<PosTaggedWord "{word}", {pos}#{tag}, {vector} at {hid}>'.format(
            word=self.text,
            pos=self.pos,
            tag=self.tag,
            vector=self.vector,
            hid=hex(id(self))
        )


class Dataset(object):
    def __init__(self):
        self.reviews = []
        self.tokenized_reviews = []
        self.pos_tagged_reviews = []
        self.language = ""

    def parse(self, filename):
        raise NotImplementedError()

    def tokenize(self):
        raise NotImplementedError()

    def pos_tag(self):
        if self.language == "ru":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            predictor = RNNMorphPredictor()
            for review in self.reviews:
                for i, sentence in enumerate(review.sentences):
                    words = [word.text for word in sentence]
                    forms = predictor.predict_sentence_tags(words)
                    pos_tagged_sentence = []
                    for word_idx, form in enumerate(forms):
                        pos_tagged_word = PosTaggedWord(sentence[word_idx], form.pos, form.tag, form.vector)
                        pos_tagged_sentence.append(pos_tagged_word)
                    review.sentences[i] = pos_tagged_sentence
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif self.language == "en":
            nlp = spacy.load('en_core_web_sm')
            for review in self.reviews:
                for i, sentence in enumerate(review.sentences):
                    text = " ".join([word.text for word in sentence])
                    doc = nlp(text)
                    tags = [(w.pos_, w.tag_, w.dep_) for w in doc]

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as w:
            w.write(jsonpickle.encode(self))

    def load(self, filename):
        assert filename.endswith('json')
        with open(filename, "r", encoding='utf-8') as f:
            dataset = jsonpickle.decode(f.read())
            self.__dict__.update(dataset.__dict__)

    def get_vocabulary(self):
        vocabulary = Vocabulary()
        for review in self.reviews:
            for sentence in review.sentences:
                vocabulary.add_sentence(" ".join([word.text.lower() for word in sentence]))
        return vocabulary

    def get_char_set(self):
        char_set = set()
        for review in self.reviews:
            for sentence in review.sentences:
                for word in sentence:
                    char_set = char_set.union(set(word.text))
        return " " + "".join(char_set)

    def get_opinion_count(self):
        return sum([len(review.aspects) for review in self.reviews])

    def get_lengths(self):
        lengths = []
        for review in self.reviews:
            lengths.append(sum([len(sentence) for sentence in review.sentences]))
        return lengths
