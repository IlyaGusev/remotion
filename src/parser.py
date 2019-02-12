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
        self.language = ""

    def parse(self, filename, **kwargs):
        raise NotImplementedError()

    def tokenize(self):
        raise NotImplementedError()

    def print_stat(self):
        raise NotImplementedError()

    def pos_tag(self):
        if self.language == "ru" or self.language == "en":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            predictor = RNNMorphPredictor(language=self.language)
            sentences = []
            for review in self.reviews:
                for i, sentence in enumerate(review.sentences):
                    words = [word.text for word in sentence]
                    sentences.append(words)
            sentences_forms = predictor.predict_sentences(sentences, 32, False)
            offset = 0
            for review in self.reviews:
                for i, sentence in enumerate(review.sentences):
                    forms = sentences_forms[offset+i]
                    for word_idx, form in enumerate(forms):
                        sentence[word_idx] = PosTaggedWord(sentence[word_idx], form.pos,
                                                           form.tag, [int(j) for j in form.vector])
                offset += len(review.sentences)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as w:
            w.write(jsonpickle.encode(self))

    def load(self, filename):
        assert filename.endswith('json')
        with open(filename, "r", encoding='utf-8') as f:
            dataset = jsonpickle.decode(f.read())
            self.__dict__.update(dataset.__dict__)

    def get_vocabulary(self, lower=True):
        vocabulary = Vocabulary()
        for review in self.reviews:
            for sentence in review.sentences:
                vocabulary.add_sentence(" ".join([word.text.lower() if lower else word.text for word in sentence]))
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


def get_dataset(filename, competition, config: DataConfig, is_train=True):
    if competition == "semeval":
        from src.semeval_parser import SemEvalDataset
        data = SemEvalDataset(language=config.language)
    elif competition == "sentirueval":
        from src.sentirueval_parser import SentiRuEvalDataset
        data = SentiRuEvalDataset()
    elif competition == "imdb":
        from src.imdb_parser import IMDBDataset
        data = IMDBDataset()
    elif competition == "sst2" or competition == "sst1":
        from src.sst_parser import SSTDataset
        data = SSTDataset()
    else:
        assert False

    if not os.path.exists("cache"):
        os.mkdir("cache")
    cache_filename = "cache/{}_{}_{}_{}.json".format(
        competition, config.language, config.domain, "train" if is_train else "test")
    if os.path.exists(cache_filename) and not config.clear_cache:
        data.load(cache_filename)
    else:
        sst_parsing = filename.endswith("txt") and (competition == "sst2" or competition == "sst1")
        imdb_parsing = filename.endswith("tsv") and competition == "imdb"
        if filename.endswith("xml") or imdb_parsing or sst_parsing:
            data.parse(filename, is_train=is_train)
            if competition != "imdb":
                data.save(cache_filename)
        elif filename.endswith("json"):
            data.load(filename)
            data.save(cache_filename)
        else:
            assert False
    data.print_stat()
    return data


def get_data(config: DataConfig, competition):
    train_data = get_dataset(config.train_filename, competition, config, True)
    test_data = get_dataset(config.test_filename, competition, config, False)
    return train_data, test_data

