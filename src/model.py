import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.functional import softmax, pad

from src.batch import VarBatch
from src.config import Config


class CNN(nn.Module):
    def __init__(self, embedding_dim, words_max_count, channels_size, kernel_sizes=(8, ), strides=(1,)):
        super(CNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.channels_size = channels_size
        self.kernel_sizes = kernel_sizes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels_size, kernel_size=(kernel_sizes[0], embedding_dim), stride=strides[0]),
            nn.ReLU()
        )
        pool_sizes = (words_max_count - kernel_sizes[0] + 1, )
        self.pool1 = nn.MaxPool2d(kernel_size=(pool_sizes[0], 1))
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(channels_size, channels_size, kernel_size=kernel_sizes[1], stride=strides[1]),
        #     nn.ReLU()
        # )
        # self.pool2 = nn.MaxPool1d(kernel_size=pool_sizes[1])
        #
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(channels_size, channels_size, kernel_size=kernel_sizes[2], stride=strides[2]),
        #     nn.ReLU()
        # )
        # self.pool3 = nn.MaxPool1d(kernel_size=pool_sizes[2])

        w = [words_max_count, ]
        for i in range(1):
            w.append(((w[i] - kernel_sizes[i]) // strides[i] + 1) // pool_sizes[i])
        self.output_size = int(w[-1] * channels_size)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, np.sqrt(2. / n))

    def forward(self, x):
        out = self.conv1(x)
        # out = out.squeeze(3)
        out = self.pool1(out)
        # out = self.conv2(out)
        # out = self.pool2(out)
        # out = self.conv3(out)
        # out = self.pool3(out)
        out = out.view(out.size(0), -1)
        return out


class RemotionRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        main_input_size = 0
        main_output_size = 0

        if config.use_word_embeddings:
            self.embedding = nn.Embedding(config.word_vocabulary_size, config.word_embedding_dim)
            self.embedding_dropout = nn.Dropout(config.word_embedding_dropout_p)
            main_input_size += config.word_embedding_dim

        if config.use_chars:
            self.char_embedding = nn.Embedding(config.char_count, config.char_embedding_dim)
            all_chars_size = config.char_max_word_length * config.char_embedding_dim
            self.char_function = nn.Linear(all_chars_size, config.char_function_output_size, bias=False)
            self.char_function_activation = nn.ReLU()
            self.char_dropout = nn.Dropout(config.char_dropout_p)
            main_input_size += config.char_function_output_size

        if config.use_pos:
            self.grammeme_dense = nn.Linear(config.gram_vector_size, config.gram_hidden_size, bias=False)
            self.grammeme_activation = nn.ReLU()
            self.grammeme_dropout = nn.Dropout(config.gram_dropout_p)
            main_input_size += config.gram_hidden_size

        if config.use_additional_features:
            main_input_size += config.additional_features_size

        if config.use_rnn:
            self.rnn = nn.LSTM(main_input_size, config.rnn_hidden_size, config.rnn_n_layers,
                               dropout=config.rnn_dropout_p, bidirectional=config.rnn_bidirectional)
            main_output_size = config.rnn_hidden_size * (2 if config.rnn_bidirectional else 1)
        elif config.use_cnn:
            self.cnn = CNN(main_input_size, self.config.word_max_count, 300)
            main_output_size = self.cnn.output_size

        self.main_output_dropout = nn.Dropout(config.rnn_output_dropout_p)

        output_layer_input_size = main_output_size
        if config.use_dense:
            dense = [nn.Linear(main_output_size, config.dense_size, bias=False) for _ in range(config.target_count)]
            self.dense_layers = nn.ModuleList(dense)
            self.dense_activation = nn.ReLU()
            self.dense_dropout_layers = nn.ModuleList([nn.Dropout(config.dense_dropout_p)
                                                       for _ in range(config.target_count)])
            output_layer_input_size = config.dense_size

        output = [nn.Linear(output_layer_input_size, config.output_size)
                  for _ in range(config.target_count)]
        self.output_layers = nn.ModuleList(output)

        if config.use_crf and config.is_sequence_predictor:
            from torchcrf import CRF
            self.crf_layers = nn.ModuleList([CRF(config.output_size) for _ in range(config.target_count)])

    def __lstm_run(self, batch: VarBatch):
        main_input = Variable(torch.FloatTensor())
        main_input = main_input.cuda() if batch.word_indices.is_cuda else main_input

        if self.config.use_word_embeddings:
            word_embeddings = self.embedding(batch.word_indices)
            word_embeddings = self.embedding_dropout(word_embeddings)
            main_input = torch.cat((main_input, word_embeddings), dim=2)

        if self.config.use_chars:
            char_embeddings = self.char_embedding(batch.char_indices)
            batch_size = char_embeddings.size(0)
            word_count = char_embeddings.size(1)
            all_chars_size = self.config.char_max_word_length * self.config.char_embedding_dim
            char_embeddings = char_embeddings.view(batch_size, word_count, all_chars_size)
            char_function_output = self.char_function_activation(self.char_function(char_embeddings))
            char_function_output = self.char_dropout(char_function_output)
            main_input = torch.cat((main_input, char_function_output), dim=2)

        if self.config.use_pos:
            grammeme = self.grammeme_activation(self.grammeme_dense(batch.gram_vectors))
            grammeme = self.grammeme_dropout(grammeme)
            main_input = torch.cat((main_input, grammeme), dim=2)

        if self.config.use_additional_features:
            main_input = torch.cat((main_input, batch.additional_features), dim=2)

        if self.config.use_rnn:
            main_input = main_input.transpose(0, 1)
            assert main_input.size(1) == batch.word_indices.size(0)
            packed = pack(main_input, batch.lengths)
            outputs, hidden = self.rnn(packed, None)
            outputs, _ = unpack(outputs)
            outputs = outputs.transpose(0, 1)

            if not self.config.is_sequence_predictor:
                hidden = hidden[0]
                n = hidden.size(0)
                outputs = torch.cat([hidden[0:n:2], hidden[1:n:2]], 2)[-1]
        elif self.config.use_cnn:
            main_input = main_input.transpose(1, 2)
            main_input = pad(main_input, (0, self.config.word_max_count - main_input.size(2)), "constant", 0)
            main_input = main_input.transpose(1, 2)
            main_input = main_input.unsqueeze(1)
            outputs = self.cnn(main_input)
        else:
            assert False

        assert outputs is not None
        outputs = self.main_output_dropout(outputs)

        predictions = []
        for i in range(self.config.target_count):
            if self.config.use_dense:
                outputs_i = self.dense_activation(self.dense_layers[i](outputs))
                outputs_i = self.dense_dropout_layers[i](outputs_i)
                predictions.append(self.output_layers[i](outputs_i))
            else:
                predictions.append(self.output_layers[i](outputs))
        if self.config.target_count == 1:
            return predictions[0]
        return predictions

    def forward(self, batch: VarBatch):
        predictions = self.__lstm_run(batch)
        y = batch.y
        if self.config.target_count == 1:
            if self.config.use_crf:
                loss = -self.crf_layers[0](predictions.transpose(0, 1), y.transpose(0, 1))
            else:
                criterion = nn.CrossEntropyLoss(size_average=False)
                if self.config.is_sequence_predictor:
                    predictions = predictions.transpose(1, 2).unsqueeze(3)
                    y = y.unsqueeze(2)
                loss = criterion(predictions, y)
        else:
            criterion = nn.CrossEntropyLoss(size_average=False)
            loss = [criterion(pred, target) for pred, target in zip(predictions, y.transpose(0, 1))]
            loss = sum(loss)
        return loss

    def predict(self, batch):
        self.eval()
        predictions = self.__lstm_run(batch)
        if self.config.target_count == 1:
            if self.config.use_crf:
                return self.crf_layers[0].decode(predictions.transpose(0, 1))
            else:
                if self.config.is_sequence_predictor:
                    return torch.argmax(softmax(predictions, dim=2), dim=2)
                else:
                    return torch.argmax(softmax(predictions, dim=1), dim=1)
        else:
            if self.config.is_sequence_predictor:
                return [torch.argmax(softmax(p, dim=2), dim=2) for p in predictions]
            else:
                return [torch.argmax(softmax(p, dim=1), dim=1) for p in predictions]


def save_model(model, optimizer, filename):
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    torch.save({
        'model': model_state_dict,
        'optimizer': optimizer.state_dict()
    }, filename)


def load_model(model_filename, config_filename, use_cuda):
    state_dict = torch.load(model_filename)
    config = Config()
    config.load(config_filename)
    model = RemotionRNN(config.model_config)
    model.load_state_dict(state_dict['model'])
    if config.model_config.use_word_embeddings:
        model.embedding.weight.requires_grad = config.train_embeddings
    model = model.cuda() if use_cuda else model

    if config.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt_lr)
    elif config.optimizer == "adadelta":
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config.opt_lr, rho=config.opt_rho, eps=config.opt_eps)
    else:
        assert False

    optimizer.load_state_dict(state_dict['optimizer'])

    return model, optimizer

