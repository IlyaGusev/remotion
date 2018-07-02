import random
import numpy as np
import torch
import torch.optim as optim

from src.model import RemotionRNN, save_model
from src.embeddings import get_embeddings
from src.batch import get_batches
from src.config import Config


def process_batch(model, batch, optimizer=None):
    if optimizer is not None:
        optimizer.zero_grad()

    loss = model.forward(batch)

    if optimizer is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    return loss.data.item()


def train_model(config_filename, train_data, vocabulary, char_set, targets,
                additionals, output_sizes):
    config = Config()
    config.load(config_filename)

    use_cuda = torch.cuda.is_available()
    print("Use cuda: ", use_cuda)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    task_key = config.competition + "-" + config.task_type

    config.model_config.word_vocabulary_size = vocabulary.size()
    config.model_config.char_count = len(char_set)

    if config.model_config.use_pos:
        gram_vector_size = len(train_data.reviews[0].sentences[0][0].vector)
        config.model_config.gram_vector_size = gram_vector_size

    config.model_config.output_size = output_sizes[task_key]
    config.save(config_filename)

    model = RemotionRNN(config.model_config)
    if config.model_config.use_word_embeddings and config.use_pretrained_embeddings:
        embeddings = get_embeddings(vocabulary, config.embeddings_filename,
                                    config.model_config.word_embedding_dim)
        model.embedding.weight = torch.nn.Parameter(embeddings, requires_grad=config.train_embeddings)
    model = model.cuda() if use_cuda else model
    print(model)

    if config.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt_lr)
    elif config.optimizer == "adadelta":
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config.opt_lr, rho=config.opt_rho, eps=config.opt_eps)
    else:
        assert False

    n = len(train_data.reviews)
    train_size = 1 - config.val_size
    border = int(n*train_size)
    reviews = random.Random(config.seed).sample(train_data.reviews, n)
    train_reviews, val_reviews = reviews[:border], reviews[border:]

    target_function = targets[task_key]
    additional_function = additionals[task_key]

    prev_val_loss = float("inf")
    bad_count = 0
    for epoch in range(config.epochs):
        train_loss = 0
        train_count = 0
        train_batches = get_batches(train_reviews, vocabulary, char_set, config.batch_size,
                                    config.model_config.word_max_count, config.model_config.char_max_word_length,
                                    target_function, additional_function, config.model_config.use_pos)
        for batch in train_batches:
            model.train()
            loss = process_batch(model, batch, optimizer)
            train_loss += loss
            train_count += 1

        val_loss = 0
        val_count = 0
        val_batches = get_batches(val_reviews, vocabulary, char_set, config.batch_size,
                                  config.model_config.word_max_count, config.model_config.char_max_word_length,
                                  target_function, additional_function, config.model_config.use_pos)
        for batch in val_batches:
            model.eval()
            loss = process_batch(model, batch, None)
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
            save_model(model, optimizer, config.model_filename)
            config.save(config_filename)
            bad_count = 0
            prev_val_loss = val_loss
        if bad_count == config.patience+1:
            break
    return model
