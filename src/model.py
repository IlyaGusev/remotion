import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchcrf import CRF

from src.batch import VarBatch
from src.config import Config

class RemotionRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        rnn_input_size = 0

        if config.use_word_embeddings:
            self.embedding = nn.Embedding(config.word_vocabulary_size, config.word_embedding_dim)
            self.embedding_dropout = nn.Dropout(config.word_embedding_dropout_p)
            rnn_input_size += config.word_embedding_dim

        if config.use_chars:
            self.char_embedding = nn.Embedding(config.char_count, config.char_embedding_dim)
            all_chars_size = config.char_max_word_length * config.char_embedding_dim
            self.char_function = nn.Linear(all_chars_size, config.char_function_output_size, bias=False)
            self.char_function_activation = nn.ReLU()
            self.char_dropout = nn.Dropout(config.char_dropout_p)
            rnn_input_size += config.char_function_output_size

        if config.use_pos:
            self.grammeme_dense = nn.Linear(config.gram_vector_size, config.gram_hidden_size, bias=False)
            self.grammeme_activation = nn.ReLU()
            self.grammeme_dropout = nn.Dropout(config.gram_dropout_p)
            rnn_input_size += config.gram_hidden_size

        if config.use_additional_features:
            rnn_input_size += config.additional_features_size

        self.rnn = nn.LSTM(rnn_input_size, config.rnn_hidden_size, config.rnn_n_layers,
                           dropout=config.rnn_dropout_p, bidirectional=config.rnn_bidirectional)
        self.rnn_output_dropout = nn.Dropout(config.rnn_output_dropout_p)

        if not config.is_multi_target:
            self.dense = nn.Linear(config.rnn_hidden_size * (2 if config.rnn_bidirectional else 1),
                                   config.dense_size, bias=False)
            self.dense_activation = nn.ReLU()
            self.dense_dropout = nn.Dropout(config.dense_dropout_p)
            self.output = nn.Linear(config.dense_size, config.output_size, bias=False)
            if config.use_crf and config.is_sequence_predictor:
                self.crf = CRF(config.output_size)
        else:
            dense = [nn.Linear(config.rnn_hidden_size * (2 if config.rnn_bidirectional else 1),
                               config.dense_size, bias=False) for _ in range(config.target_count)]
            self.dense = nn.ModuleList(dense)
            self.dense_activation = nn.ReLU()
            self.dense_dropout = nn.ModuleList([nn.Dropout(config.dense_dropout_p) for _ in range(config.target_count)])
            output = [nn.Linear(config.dense_size, config.output_size, bias=False) for _ in range(config.target_count)]
            self.output = nn.ModuleList(output)

    def __lstm_run(self, batch: VarBatch):
        rnn_input = Variable(torch.FloatTensor())
        rnn_input = rnn_input.cuda() if batch.word_indices.is_cuda else rnn_input
        if self.config.use_word_embeddings:
            word_embeddings = self.embedding(batch.word_indices)
            word_embeddings = self.embedding_dropout(word_embeddings)
            rnn_input = torch.cat((rnn_input, word_embeddings), dim=2)
        if self.config.use_chars:
            char_embeddings = self.char_embedding(batch.char_indices)
            batch_size = char_embeddings.size(0)
            word_count = char_embeddings.size(1)
            all_chars_size = self.config.char_max_word_length * self.config.char_embedding_dim
            char_embeddings = char_embeddings.view(batch_size, word_count, all_chars_size)
            char_function_output = self.char_function_activation(self.char_function(char_embeddings))
            char_function_output = self.char_dropout(char_function_output)
            rnn_input = torch.cat((rnn_input, char_function_output), dim=2)
        if self.config.use_pos:
            grammeme = self.grammeme_activation(self.grammeme_dense(batch.gram_vectors))
            grammeme = self.grammeme_dropout(grammeme)
            rnn_input = torch.cat((rnn_input, grammeme), dim=2)
        if self.config.use_additional_features:
            rnn_input = torch.cat((rnn_input, batch.additional_features), dim=2)

        rnn_input = rnn_input.transpose(0, 1)
        assert rnn_input.size(1) == batch.word_indices.size(0)
        packed = pack(rnn_input, batch.lengths)
        outputs, hidden = self.rnn(packed, None)
        outputs, _ = unpack(outputs)
        outputs = outputs.transpose(0, 1)

        if self.config.is_sequence_predictor:
            outputs = self.rnn_output_dropout(outputs)
        else:
            hidden = hidden[0]
            n = hidden.size(0)
            outputs = torch.cat([hidden[0:n:2], hidden[1:n:2]], 2)[-1]
            # outputs = self.rnn_output_dropout(hidden)

        if not self.config.is_multi_target:
            outputs = self.dense_activation(self.dense(outputs))
            outputs = self.dense_dropout(outputs)
            predictions = self.output(outputs)
        else:
            predictions = []
            for i in range(self.config.target_count):
                outputs_i = self.dense_activation(self.dense[i](outputs))
                outputs_i = self.dense_dropout[i](outputs_i)
                predictions.append(self.output[i](outputs_i))
        return predictions

    def forward(self, batch: VarBatch):
        predictions = self.__lstm_run(batch)
        y = batch.y
        if not self.config.is_multi_target:
            if self.config.use_crf:
                loss = -self.crf(predictions.transpose(0, 1), y.transpose(0, 1))
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
        if not self.config.is_multi_target:
            if self.config.use_crf:
                return self.crf.decode(predictions.transpose(0, 1))
            else:
                if self.config.is_sequence_predictor:
                    return torch.argmax(nn.functional.softmax(predictions, dim=2), dim=2)
                else:
                    return torch.argmax(nn.functional.softmax(predictions, dim=1), dim=1)
        else:
            if self.config.is_sequence_predictor:
                return [torch.argmax(nn.functional.softmax(pred, dim=2), dim=2) for pred in predictions]
            else:
                return [torch.argmax(nn.functional.softmax(pred, dim=1), dim=1) for pred in predictions]


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
        model.embedding.weight.requires_grad = False
    model = model.cuda() if use_cuda else model

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer.load_state_dict(state_dict['optimizer'])

    return model, optimizer

