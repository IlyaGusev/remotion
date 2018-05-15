import torch
import torch.nn as nn
import torch.optim as optim

class RemotionRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, rnn_size, gram_vector_size, gram_hidden_size, n_layers=3,
                 dropout=0.3, output_size=3,bidirectional=True):
        super(RemotionRNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.gram_vector_size = gram_vector_size
        self.gram_hidden_size = gram_hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.grammeme_dense = nn.Linear(gram_vector_size, gram_hidden_size)
        self.grammeme_activation = nn.ReLU()
        self.rnn = nn.LSTM(embedding_dim + gram_hidden_size, rnn_size, n_layers,
                           dropout=dropout, bidirectional=bidirectional)
        self.output = nn.Linear(rnn_size * (2 if bidirectional else 1), output_size)

    def forward(self, input_seqs, gram_vectors, hidden=None):
        embedded = self.embedding(input_seqs)
        grammeme = self.grammeme_activation(self.grammeme_dense(gram_vectors))
        rnn_input = torch.cat((embedded, grammeme), dim=2)
        outputs, hidden = self.rnn(rnn_input, hidden)
        predictions = self.output(outputs).squeeze(2)
        return predictions

def save_model(model, optimizer, filename):
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    torch.save({
        'model': model_state_dict,
        'input_size': model.input_size,
        'n_layers': model.n_layers,
        'rnn_size': model.rnn_size,
        'dropout': model.dropout,
        'embedding_dim': model.embedding_dim,
        'gram_vector_size': model.gram_vector_size,
        'gram_hidden_size': model.gram_hidden_size,
        'bidirectional': model.bidirectional,
        'optimizer': optimizer.state_dict()
    }, filename)


def load_model(model_filename, use_cuda):
    state_dict = torch.load(model_filename)
    model = RemotionRNN(
        input_size=state_dict['input_size'],
        embedding_dim=state_dict['embedding_dim'],
        rnn_size=state_dict['rnn_size'],
        gram_vector_size=state_dict['gram_vector_size'],
        gram_hidden_size=state_dict['gram_hidden_size'],
        n_layers=state_dict['n_layers'],
        bidirectional=state_dict['bidirectional'],
        dropout=state_dict['dropout']
    )
    model.load_state_dict(state_dict['model'])
    model.embedding.weight.requires_grad = False
    model = model.cuda() if use_cuda else model

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer.load_state_dict(state_dict['optimizer'])

    return model, optimizer
