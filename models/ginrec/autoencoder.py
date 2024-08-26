import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, entity_dim, layer_dims, use_activation=True, tanh_range=False):
        super().__init__()
        n_layers = len(layer_dims)
        self.use_activation = use_activation
        self.tanh_range = tanh_range

        self.encoders = nn.ParameterList()
        self.decoders = nn.ParameterList()
        self.encoder_bias = nn.ParameterList()
        self.decoder_bias = nn.ParameterList()
        out_dim = entity_dim

        for layer in range(n_layers):
            in_dim = out_dim
            out_dim = layer_dims[layer]

            decode_out = out_dim
            e_l = nn.Parameter(torch.Tensor(in_dim, out_dim))
            d_l = nn.Parameter(torch.Tensor(decode_out, in_dim))
            e_b = nn.Parameter(torch.Tensor(out_dim,))
            d_b = nn.Parameter(torch.Tensor(in_dim,))

            nn.init.xavier_uniform_(e_l)
            nn.init.xavier_uniform_(d_l)
            nn.init.constant_(e_b, 0.0)
            nn.init.constant_(d_b, 0.0)

            self.encoders.append(e_l)
            self.decoders.append(d_l)
            self.encoder_bias.append(e_b)
            self.decoder_bias.append(d_b)

        self.decoders = nn.ParameterList(reversed(self.decoders))
        self.decoder_bias = nn.ParameterList(reversed(self.decoder_bias))

        self.activation = nn.Tanh()

        self.loss_fn = nn.MSELoss(reduction='none')

    def propagate(self, embeddings, mode):
        if mode == 'encode':
            iterator = list(zip(self.encoders, self.encoder_bias))
        elif mode == 'decode':
            iterator = list(zip(self.decoders, self.decoder_bias))
        else:
            raise ValueError('Invalid mode')

        length = len(iterator)
        for i, (w, b) in enumerate(iterator, 1):
            embeddings = torch.matmul(embeddings, w) + b

            if self.use_activation:
                if mode == 'decode' and i == length:
                    embeddings = embeddings
                else:
                    embeddings = self.activation(embeddings)

        return embeddings

    def forward(self, embeddings):
        encoded = self.propagate(embeddings, 'encode')
        decoded = self.propagate(encoded, 'decode')
        return encoded, decoded

    def loss(self, target, decoded):
        loss = self.loss_fn(decoded, target)

        return loss