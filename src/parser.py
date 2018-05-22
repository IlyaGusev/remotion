import os
import jsonpickle
from collections import namedtuple

import spacy
from rnnmorph.predictor import RNNMorphPredictor

from src.vocabulary import Vocabulary
from src.config import DataConfig

WordFormOut = namedtuple("WordFormOut", "pos tag normal_form vector")


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


class SpacyVectorizer(object):
    def __init__(self):
        self.pos = set()
        self.tags = set()
        self.deps = set()

    def add_sentence(self, text, model):
        doc = model(text)
        for word in doc:
            self.pos.add(word.pos_)
            self.tags.add(word.tag_)
            self.deps.add(word.dep_)

    def get_forms(self, text, model):
        doc = model(text)
        forms = []
        for word in doc:
            pos_part = self.__vectorize(self.pos, word.pos_)
            tag_part = self.__vectorize(self.tags, word.tag_)
            dep_part = self.__vectorize(self.deps, word.dep_)
            vector = pos_part + tag_part + dep_part
            forms.append(WordFormOut(pos=word.pos_, tag=word.tag_, normal_form=word.lemma_, vector=vector))
        return forms

    def is_empty(self):
        return not (self.pos or self.tags or self.deps)

    @staticmethod
    def __vectorize(collection, element):
        vector = [0 for _ in range(len(collection))]
        l = list(sorted(collection))
        if element in l:
            vector[l.index(element)] = 1
        return vector

    def save(self, dump_filename) -> None:
        with open(dump_filename, "w") as f:
            f.write(jsonpickle.encode(self, f))

    def load(self, dump_filename):
        with open(dump_filename, "r") as f:
            vectorizer = jsonpickle.decode(f.read())
            self.__dict__.update(vectorizer.__dict__)


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

    def pos_tag(self, vectorizer_path=None):
        if self.language == "ru":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            predictor = RNNMorphPredictor()
            for review in self.reviews:
                for i, sentence in enumerate(review.sentences):
                    words = [word.text for word in sentence]
                    forms = predictor.predict_sentence_tags(words)
                    for word_idx, form in enumerate(forms):
                        sentence[word_idx] = PosTaggedWord(sentence[word_idx], form.pos, form.tag, form.vector)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        elif self.language == "en":
            model = spacy.load('en_core_web_sm')
            vectorizer = SpacyVectorizer()
            if vectorizer_path is None or not os.path.exists(vectorizer_path):
                for review in self.reviews:
                    for sentence in review.parsed_sentences:
                        vectorizer.add_sentence(sentence.text, model)
                if vectorizer_path is not None:
                    vectorizer.save(vectorizer_path)
            else:
                vectorizer.load(vectorizer_path)
            for review in self.reviews:
                for sentence in review.sentences:
                    text = " ".join([word.text for word in sentence])
                    forms = vectorizer.get_forms(text, model)
                    for i, (word, form) in enumerate(zip(sentence, forms)):
                        sentence[i] = PosTaggedWord(word, form.pos, form.tag, form.vector)

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

    def get_aspect_categories(self):
        return {}

    def get_review_categories(self):
        return {}

def get_dataset(filename, competition, language, domain, is_train=True, vectorizer_path=None):
    if competition == "semeval":
        from src.semeval_parser import SemEvalDataset
        data = SemEvalDataset(language=language)
    elif competition == "sentirueval":
        from src.sentirueval_parser import SentiRuEvalDataset
        data = SentiRuEvalDataset()
    else:
        assert False
    if filename.endswith("xml"):
        data.parse(filename, vectorizer_path)
        data.save("{}_{}_{}_{}.json".format(competition, language, domain, "train" if is_train else "test"))
    elif filename.endswith("json"):
        data.load(filename)
    else:
        assert False
    print("Num of reviews: " + str(len(data.reviews)))
    print("Num of opinions: " + str(data.get_opinion_count()))
    print("Max review length: " + str(max(data.get_lengths())))
    print(data.reviews[0].sentences[0])
    print(data.reviews[0].sentences[0])
    return data

def get_data(config: DataConfig):
    train_data = get_dataset(config.train_filename, config.competition, config.language, config.domain, True, config.vectorizer_path)
    test_data = get_dataset(config.test_filename, config.competition, config.language, config.domain, False, config.vectorizer_path)
    return train_data, test_data

