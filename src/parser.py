import os
import jsonpickle
from rnnmorph.predictor import RNNMorphPredictor
from rnnmorph.data_preparation.grammeme_vectorizer import GrammemeVectorizer

from src.vocabulary import Vocabulary

class Word(object):
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

class PosTaggedWord(Word):
    def __init__(self, word, pos, tag, vector):
        Word.__init__(self, word.text, word.begin, word.end)
        self.opinion = word.opinion
        self.pos = pos
        self.tag = tag
        self.vector = vector

    def __repr__(self):
        return '<PosTaggedWord "{word}", {pos}#{tag}, {vector} at {hid}>'.format(
            word = self.text,
            pos = self.pos,
            tag = self.tag,
            vector = self.vector,
            hid = hex(id(self))
        )

class Dataset(object):
    def __init__(self):
        self.reviews = []
        self.tokenized_reviews = []
        self.pos_tagged_reviews = []

    def parse(self, filename, grammeme_vectorizer_path):
        raise NotImplementedError()

    def tokenize(self):
        raise NotImplementedError()

    def get_vocabulary(self):
        vocabulary = Vocabulary()
        for review in self.tokenized_reviews:
            for sentence in review:
                vocabulary.add_sentence(" ".join([word.text for word in sentence]))
        return vocabulary

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as w:
            w.write(jsonpickle.encode(self))

    def load(self, filename):
        assert filename.endswith('json')
        with open(filename, "r", encoding='utf-8') as f:
            dataset = jsonpickle.decode(f.read())
            self.__dict__.update(dataset.__dict__)

    def get_opinion_count(self):
        return sum([len(review.aspects) for review in self.reviews])

    def get_lengths(self):
        lengths = []
        for review in self.tokenized_reviews:
            lengths.append(sum([len(sentence) for sentence in review]))
        return lengths

    def get_colored_rate(self):
        colored_words_count = 0
        words_count = 0
        for review in self.tokenized_reviews:
            for sentence in review:
                for word in sentence:
                    words_count += 1
                    if word.is_colored():
                        colored_words_count += 1
        return float(colored_words_count)/words_count

    def pos_tag(self, grammeme_vectorizer_path):
        if grammeme_vectorizer_path is not None:
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
                        pos_tagged_word = PosTaggedWord(sentence[word_idx], form.pos, form.tag, vector)
                        pos_tagged_sentence.append(pos_tagged_word)
                    pos_tagged_reviews[-1].append(pos_tagged_sentence)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return pos_tagged_reviews
        raise NotImplementedError()

