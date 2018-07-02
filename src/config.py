import json
import copy
import os


class ModelConfig(object):
    def __init__(self):
        self.is_sequence_predictor = True
        self.use_crf = False
        self.use_pos = True
        self.use_chars = False
        self.use_word_embeddings = True
        self.use_additional_features = True
        self.use_rnn = True
        self.use_cnn = False
        self.use_dense = True

        self.target_count = 1
        self.additional_features_size = 1

        self.word_vocabulary_size = 2
        self.word_embedding_dim = 500
        self.word_embedding_dropout_p = 0.2
        self.word_max_count = 1500

        self.rnn_n_layers = 2
        self.rnn_hidden_size = 50
        self.rnn_dropout_p = 0.5
        self.rnn_bidirectional = True
        self.rnn_output_dropout_p = 0.3

        self.gram_vector_size = 52
        self.gram_hidden_size = 30
        self.gram_dropout_p = 0.2

        self.char_count = 50
        self.char_embedding_dim = 4
        self.char_function_output_size = 50
        self.char_dropout_p = 0.2
        self.char_max_word_length = 30

        self.dense_size = 32
        self.dense_dropout_p = 0.3

        self.output_size = 3

        self.assert_correctness()

    def assert_correctness(self):
        assert not self.use_crf or self.use_crf and self.is_sequence_predictor
        assert self.target_count == 1 or self.target_count != 1 and not self.use_crf
        assert self.use_rnn and not self.use_cnn or self.use_cnn and not self.use_rnn


class DataConfig(object):
    def __init__(self):
        self.language = "en"
        self.domain = "rest"
        self.train_filename = ""
        self.test_filename = ""
        self.clear_cache = False


class Config(object):
    def __init__(self):
        self.data_config = DataConfig()
        self.model_config = ModelConfig()

        self.competition = "semeval"
        self.task_type = "a"
        self.result_type = "xml"
        self.experiment = "base"

        self.val_size = 0.2
        self.epochs = 100
        self.batch_size = 8
        self.patience = 2
        self.optimizer = "adam"
        self.opt_lr = 0.001
        self.opt_rho = 0.95
        self.opt_eps = 1e-6

        self.lower_vocabulary = True

        self.use_pretrained_embeddings = True
        self.train_embeddings = False
        self.embeddings_filename = ""
        self.seed = 42

        self.assert_correctness()

    def assert_correctness(self):
        assert self.competition in ["semeval", "sentirueval", "imdb", "sst2", "sst1"]
        assert self.optimizer in ["adam", "adadelta"]
        assert self.result_type in ["xml", "csv", "txt"]

    @property
    def output_filename(self):
        if not os.path.exists("submissions"):
            os.mkdir("submissions")
        return "submissions/{}_{}_{}_{}_{}.{}".format(
            self.competition, self.data_config.language, self.data_config.domain, self.task_type,
            self.experiment, self.result_type)

    @property
    def model_filename(self):
        if not os.path.exists("models"):
            os.mkdir("models")
        return "models/{}_{}_{}_{}_{}.pt".format(
            self.competition, self.data_config.language, self.data_config.domain, self.task_type,
            self.experiment)

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            d = copy.deepcopy(self.__dict__)
            d['model_config'] = self.model_config.__dict__
            d['data_config'] = self.data_config.__dict__
            f.write(json.dumps(d, sort_keys=True, indent=4)+"\n")

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            d = json.loads(f.read())
            self.__dict__.update(d)
            self.model_config = ModelConfig()
            self.model_config.__dict__.update(d['model_config'])
            self.model_config.assert_correctness()

            self.data_config = DataConfig()
            self.data_config.__dict__.update(d['data_config'])

            self.assert_correctness()


def get_targets_additionals(train_data):
    aspect_categories = train_data.get_aspect_categories()
    rev_aspect_categories = {value: key for key, value in aspect_categories.items()}
    print("Aspect categories: ", len(aspect_categories))

    review_categories = train_data.get_review_categories()
    print("Review categories: ", len(review_categories))

    def get_target_func_from_word_func(word_func):
        def target_func(review):
            return [word_func(word) for sentence in review.sentences for word in sentence]
        return target_func

    def sentirueval_word_function_a(word):
        for opinion in word.opinions:
            if opinion.type != 0 or opinion.mark != 0:
                continue
            if opinion.words[0].text == word.text:
                return 1
            return 2
        return 0

    def sentirueval_word_function_b(word):
        for opinion in word.opinions:
            if opinion.mark != 0:
                continue
            opinion_type = opinion.type
            if opinion.words[0].text == word.text:
                return 2 * opinion_type + 1
            return 2 * opinion_type + 2
        return 0

    def polarity_word_function(word):
        for opinion in word.opinions:
            return opinion.polarity + 1
        return 0

    def sentirueval_word_function_d(word):
        for opinion in word.opinions:
            return aspect_categories[opinion.category] + 1
        return 0

    def sentirueval_target_function_e(review):
         values = []
         for key in sorted(review_categories.keys()):
             values.append(review.categories[key])
         return values[0]

    def semeval_word_function_12(word):
        for opinion in word.opinions:
            opinion_category = aspect_categories[opinion.cat_first+"#"+opinion.cat_second]
            if opinion.words[0].text == word.text:
                return 2 * opinion_category + 1
            return 2 * opinion_category + 2
        return 0 

    def opinion_additional_function(word):
        if word.opinions:
            return [len(word.opinions)]
        return [0]

    def clf_target_function(review):
        return review.sentiment

    targets = {
        'semeval-12': get_target_func_from_word_func(semeval_word_function_12),
        'semeval-3': get_target_func_from_word_func(polarity_word_function),
        'sentirueval-a': get_target_func_from_word_func(sentirueval_word_function_a),
        'sentirueval-b': get_target_func_from_word_func(sentirueval_word_function_b),
        'sentirueval-c': get_target_func_from_word_func(polarity_word_function),
        'sentirueval-d': get_target_func_from_word_func(sentirueval_word_function_d),
        'sentirueval-e': sentirueval_target_function_e,
        'imdb-clf': clf_target_function,
        'sst1-clf': clf_target_function,
        'sst2-clf': clf_target_function,
    }

    additionals = {
        'semeval-12': None,
        'semeval-3': opinion_additional_function,
        'sentirueval-a': None,
        'sentirueval-b': None,
        'sentirueval-c': opinion_additional_function,
        'sentirueval-d': opinion_additional_function,
        'sentirueval-e': None,
        'imdb-clf': None,
        'sst1-clf': None,
        'sst2-clf': None
    }

    output_sizes = {
        'semeval-1': 2 * len(aspect_categories) + 1,
        'semeval-12': 2 * len(aspect_categories) + 1,
        'semeval-3': 5,
        'sentirueval-a': 3,
        'sentirueval-b': 7,
        'sentirueval-c': 5,
        'sentirueval-d': 2 * len(aspect_categories) + 1,
        'imdb-clf': 2,
        'sst1-clf': 5,
        'sst2-clf': 2
    }

    return targets, additionals, rev_aspect_categories, output_sizes
