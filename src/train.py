import torch
import torch.optim as optim
from torch.autograd import Variable
import random
import copy
from sklearn.metrics import roc_auc_score
from torchcrf import CRF

from src.model import Config, RemotionRNN, save_model, load_model
from src.embeddings import get_embeddings
from src.metrics import choose_threshold_by_f1

def get_batches(data, vocabulary, char_set, gram_vector_size, batch_size,
        max_length, max_word_length, target_function):
    text_batch = []
    gram_batch = []
    char_batch = []
    y = []

    for review in data:
        indices = []
        gram_vectors = []
        target = []
        for sentence in review:
            words_texts = [word.text for word in sentence]
            chars = [[char_set.find(ch) if char_set.find(ch) != -1 else 0 for ch in word] for word in words_texts]
            text = " ".join([text.lower() for text in words_texts])
            indices += vocabulary.get_indices(text)
            gram_vectors += [word.vector for word in sentence]
            target += [target_function(word) for word in sentence]

        indices = vocabulary.pad_indices(indices, max_length)[:max_length]
        for _ in range(max_length-len(gram_vectors)):
            gram_vectors.append([0 for _ in range(gram_vector_size)])
        gram_vectors = gram_vectors[:max_length]
        chars = [word + [0 for _ in range(max_word_length-len(word))] for word in chars]
        chars += [[0 for _ in range(max_word_length)] for _ in range(max_length - len(chars))]
        target += [0 for _ in range(max_length-len(target))]
        target = target[:max_length]

        text_batch.append(indices)
        gram_batch.append(gram_vectors)
        char_batch.append(chars)
        y.append(target)

        if len(text_batch) == batch_size:
            text_var = Variable(torch.LongTensor(text_batch), requires_grad=False)
            gram_var = Variable(torch.FloatTensor(gram_batch), requires_grad=False)
            char_var = Variable(torch.LongTensor(char_batch), requires_grad=False)
            y_var = Variable(torch.LongTensor(y), requires_grad=False)
            yield text_var, gram_var, char_var, y_var
            text_batch = []
            gram_batch = []
            char_batch = []
            y = []
    if len(text_batch) != 0:
        text_var = Variable(torch.LongTensor(text_batch), requires_grad=False)
        gram_var = Variable(torch.FloatTensor(gram_batch), requires_grad=False)
        char_var = Variable(torch.LongTensor(char_batch), requires_grad=False)
        y_var = Variable(torch.LongTensor(y), requires_grad=False)
        yield text_var, gram_var, char_var, y_var

def train_batch(model, text_batch, gram_batch, char_batch, y, use_cuda, optimizer=None):
    text_batch = text_batch.cuda() if use_cuda else text_batch
    gram_batch = gram_batch.cuda() if use_cuda else gram_batch
    char_batch = char_batch.cuda() if use_cuda else char_batch
    y = y.cuda() if use_cuda else y 

    if optimizer is not None:
        optimizer.zero_grad()

    loss = model.forward(text_batch, gram_batch, char_batch, y)

    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    return loss.data.item()

def predict_batch(model, text_batch, gram_batch, char_batch, use_cuda):
    text_batch = text_batch.cuda() if use_cuda else text_batch
    gram_batch = gram_batch.cuda() if use_cuda else gram_batch
    char_batch = char_batch.cuda() if use_cuda else char_batch
    model.eval()
    return model.predict(text_batch, gram_batch, char_batch)

def train_model(config_filename,
                model_filename,
                train_data,
                vocabulary,
                char_set,
                target_function,
                use_pretrained_embeddings=True,
                embeddings_filename="w2v.txt",
                val_size=0.2,
                lr=0.001,
                seed=42,
                batch_size=8,
                max_length=300,
                max_word_length=30,
                epochs=15,
                patience=1):
    config = Config()
    config.load(config_filename)
    use_cuda = torch.cuda.is_available()

    config.word_vocabulary_size = vocabulary.size()
    config.char_count = len(char_set)
    gram_vector_size = len(train_data[0][0][0].vector)
    config.gram_vector_size = gram_vector_size
    config.char_max_word_length = max_word_length

    model = RemotionRNN(config)
    if use_pretrained_embeddings:
        embeddings = get_embeddings(vocabulary, embeddings_filename)
        model.embedding.weight = torch.nn.Parameter(embeddings, requires_grad=False)
    print(model)
    model = model.cuda() if use_cuda else model

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_size = 1 - val_size
    border = int(len(train_data)*train_size)
    train_data = copy.deepcopy(train_data)
    random.Random(seed).shuffle(train_data)
    train_data, val_data = train_data[:border], train_data[border:]

    prev_val_loss = float("inf")
    bad_count = 0
    for epoch in range(epochs):
        train_loss = 0
        train_count = 0
        train_batches = get_batches(train_data, vocabulary, char_set,
            gram_vector_size, batch_size, max_length, max_word_length, target_function)
        for text_batch, gram_batch, char_batch, y in train_batches:
            model.train()
            loss = train_batch(model, text_batch, gram_batch, char_batch, y, use_cuda, optimizer)
            train_loss += loss
            train_count += 1

        val_loss = 0
        val_count = 0
        val_batches = get_batches(val_data, vocabulary, char_set,
            gram_vector_size, batch_size, max_length, max_word_length, target_function)
        for text_batch, gram_batch, char_batch, y in val_batches:
            model.eval()
            loss = train_batch(model, text_batch, gram_batch, char_batch, y, use_cuda, None)
            val_loss += loss
            val_count += 1

        print("Epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}".format(
            epoch = epoch,
            train_loss = train_loss/train_count,
            val_loss = val_loss/val_count
        ))
        if prev_val_loss < val_loss:
            bad_count += 1
        else:
            save_model(model, optimizer, model_filename, config_filename)
            bad_count = 0
            prev_val_loss = val_loss
        if bad_count == patience+1:
            break
    return model
