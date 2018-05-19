import random
import copy

import torch
import torch.optim as optim

from src.model import Config, RemotionRNN, save_model
from src.embeddings import get_embeddings
from src.batch import get_batches


def train_batch(model, batch, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()

    loss = model.forward(batch)

    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    return loss.data.item()


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
    print("Use cuda: ", use_cuda)

    config.word_vocabulary_size = vocabulary.size()
    config.char_count = len(char_set)
    gram_vector_size = len(train_data[0][0][0].vector)
    config.gram_vector_size = gram_vector_size
    config.char_max_word_length = max_word_length

    model = RemotionRNN(config)
    if config.use_word_embeddings and use_pretrained_embeddings:
        embeddings = get_embeddings(vocabulary, embeddings_filename, config.word_embedding_dim)
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
        train_batches = get_batches(train_data, vocabulary, char_set, batch_size,
                                    max_length, max_word_length, target_function)
        for batch in train_batches:
            model.train()
            loss = train_batch(model, batch, optimizer)
            train_loss += loss
            train_count += 1

        val_loss = 0
        val_count = 0
        val_batches = get_batches(val_data, vocabulary, char_set, batch_size,
                                  max_length, max_word_length, target_function)
        for batch in val_batches:
            model.eval()
            loss = train_batch(model, batch, None)
            val_loss += loss
            val_count += 1

        print("Epoch: {epoch}, train loss: {train_loss}, val loss: {val_loss}".format(
            epoch=epoch,
            train_loss=train_loss/train_count,
            val_loss=val_loss/val_count
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
