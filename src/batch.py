import torch
from torch.autograd import Variable


class Batch:
    def __init__(self):
        self.word_indices = []
        self.gram_vectors = []
        self.char_indices = []
        self.additional_features = []
        self.y = []
        self.reset()

    def reset(self):
        self.word_indices = []
        self.gram_vectors = []
        self.char_indices = []
        self.additional_features = []
        self.y = []

    def add_review(self, review, vocabulary, char_set, max_length, max_word_length,
                   target_function, additional_function):
        gram_vector_size = len(review[0][0].vector)
        indices = []
        gram_vectors = []
        target = []
        chars = []
        additional = []

        for sentence in review:
            words_texts = [word.text for word in sentence]
            chars += [[char_set.find(ch) if char_set.find(ch) != -1 else 0 for ch in word] for word in words_texts]
            text = " ".join([text.lower() for text in words_texts])
            indices += vocabulary.get_indices(text)
            gram_vectors += [word.vector for word in sentence]
            target += [target_function(word) for word in sentence]
            if additional_function is not None:
                additional += [additional_function(word) for word in sentence]

        indices = vocabulary.pad_indices(indices, max_length)[:max_length]

        for _ in range(max_length - len(gram_vectors)):
            gram_vectors.append([0 for _ in range(gram_vector_size)])
        gram_vectors = gram_vectors[:max_length]

        chars = [word + [0 for _ in range(max_word_length - len(word))] for word in chars]
        chars += [[0 for _ in range(max_word_length)] for _ in range(max_length - len(chars))]

        target += [0 for _ in range(max_length - len(target))]
        target = target[:max_length]

        additional += [[0] for _ in range(max_length - len(additional))]
        additional = additional[:max_length]

        self.word_indices.append(indices)
        self.gram_vectors.append(gram_vectors)
        self.char_indices.append(chars)
        self.additional_features.append(additional)
        self.y.append(target)

    def size(self):
        return len(self.word_indices)


class VarBatch:
    def __init__(self, batch: Batch):
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        self.word_indices = Variable(LongTensor(batch.word_indices), requires_grad=False)
        self.gram_vectors = Variable(FloatTensor(batch.gram_vectors), requires_grad=False)
        self.char_indices = Variable(LongTensor(batch.char_indices), requires_grad=False)
        self.additional_features = Variable(FloatTensor(batch.additional_features), requires_grad=False)
        self.y = Variable(LongTensor(batch.y), requires_grad=False)


def get_batches(data, vocabulary, char_set, batch_size,
                max_length, max_word_length, target_function, additional_function=None):
    batch = Batch()
    for review in data:
        batch.add_review(review, vocabulary, char_set, max_length, max_word_length,
                         target_function, additional_function)

        if batch.size() == batch_size:
            yield VarBatch(batch)
            batch.reset()
    if batch.size() != 0:
        yield VarBatch(batch)
