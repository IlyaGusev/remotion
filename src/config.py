import json
import copy


class ModelConfig(object):
    def __init__(self):
        self.is_sequence_predictor = True
        self.use_crf = False
        self.use_pos = True
        self.use_chars = False
        self.use_word_embeddings = True
        self.use_additional_features = True
        self.is_multi_target = False
        self.target_count = 5
        self.additional_features_size = 1
        self.word_vocabulary_size = 2
        self.word_embedding_dim = 500
        self.word_embedding_dropout_p = 0.2
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


class DataConfig(object):
    def __init__(self):
        self.language = "en"
        self.domain = "rest"
        self.train_filename = ""
        self.test_filename = ""
        self.vectorizer_path = "vectorizer.json"
        self.clear_cache = False


class Config(object):
    def __init__(self):
        self.competition = "semeval"
        self.task_type = "a"
        self.val_size = 0.2
        self.epochs = 100
        self.lr = 0.001
        self.batch_size = 8
        self.patience = 2
        self.max_length = 300
        self.use_pretrained_embedding = True
        self.embeddings_filename = ""
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.output_filename = "submission.xml"
        self.seed = 42
        self.model_filename = "model.pt"

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
            self.data_config = DataConfig()
            self.data_config.__dict__.update(d['data_config'])


def get_targets_additionals(train_data):
    aspect_categories = train_data.get_aspect_categories()
    rev_aspect_categories = {value: key for key, value in aspect_categories.items()}
    print("Aspect categories: ", len(aspect_categories))

    review_categories = train_data.get_review_categories()
    rev_review_categories = {value:key for key, value in review_categories.items()}
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

    targets = {
        'semeval-12': get_target_func_from_word_func(semeval_word_function_12),
        'semeval-3': get_target_func_from_word_func(polarity_word_function),
        'sentirueval-a': get_target_func_from_word_func(sentirueval_word_function_a),
        'sentirueval-b': get_target_func_from_word_func(sentirueval_word_function_b),
        'sentirueval-c': get_target_func_from_word_func(polarity_word_function),
        'sentirueval-d': get_target_func_from_word_func(sentirueval_word_function_d),
        'sentirueval-e': sentirueval_target_function_e
    }

    additionals = {
        'semeval-12': None,
        'semeval-3': opinion_additional_function,
        'sentirueval-a': None,
        'sentirueval-b': None,
        'sentirueval-c': opinion_additional_function,
        'sentirueval-d': opinion_additional_function,
        'sentirueval-e': None
    }

    output_sizes = {
        'semeval-1': 2 * len(aspect_categories) + 1,
        'semeval-12': 2 * len(aspect_categories) + 1,
        'semeval-3': 5,
        'sentirueval-a': 3,
        'sentirueval-b': 7,
        'sentirueval-c': 5,
        'sentirueval-d': 2 * len(aspect_categories) + 1
    }

    return targets, additionals, rev_aspect_categories, output_sizes
