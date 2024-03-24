import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.distributions.categorical import Categorical


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(dataset.vocab_size, embed_size, padding_idx=self.dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, dataset.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """

        embeddings = self.embedding(indices)

        rnn_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths=lengths, enforce_sorted=False, batch_first=True)

        rnn_output, _ = self.rnn(rnn_input)

        linear_input, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        logits = self.linear(linear_input)

        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """

        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)

        tokens = torch.tensor(tokens).unsqueeze(0)

        embeddings = self.embedding(tokens)

        rnn_output, hidden = self.rnn(embeddings)

        logits = self.linear(rnn_output)

        logits /= temp
        next_token = Categorical(logits=logits[:, -1:]).sample()
        

        tokens = torch.cat([tokens, next_token], dim=1)

        for _ in range(self.max_length - len(tokens)):
            embeddings = self.embedding(tokens)

            rnn_output, hidden = self.rnn(embeddings, hidden)

            logits = self.linear(rnn_output)

            logits /= temp
            next_token = Categorical(logits=logits[:, -1:]).sample()

            if next_token == self.dataset.eos_id:
                break

            tokens = torch.cat([tokens, next_token], dim=1)

        return self.dataset.ids2text(tokens.squeeze())
