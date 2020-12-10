from parlai.zoo.bert.build import download
from parlai.utils.misc import warn_once
from parlai.core.torch_agent import History

import parlai.core.torch_generator_agent as tga
from parlai.core.torch_agent import Output

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from os.path import join

# from .helpers import VOCAB_PATH

from parlai.agents.turntaking.turntaking_dictionary import TurnTakingDictionaryAgent


##########################################################################
# From ParlAI Example
##########################################################################
class ExampleModel(tga.TorchGeneratorModel):
    def __init__(self, dictionary, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)

    def output(self, decoder_output):
        return F.linear(decoder_output, self.embeddings.weight)

    def reorder_encoder_states(self, encoder_states, indices):
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]


class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super().__init__()
        _vocab_size, esz = embeddings.weight.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=esz, hidden_size=hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, input_tokens):
        embedded = self.embeddings(input_tokens)
        _output, hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super().__init__()
        _vocab_size, self.esz = embeddings.weight.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=self.esz, hidden_size=hidden_size, num_layers=1, batch_first=True
        )

    def forward(self, input, encoder_state, incr_state=None):
        embedded = self.embeddings(input)
        if incr_state is None:
            state = encoder_state
        else:
            state = incr_state
        output, incr_state = self.lstm(embedded, state)
        return output, incr_state


##########################################################################


class TurntakingAgent(tga.TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        super(TurntakingAgent, cls).add_cmdline_args(argparser)
        group = argparser.add_argument_group("TurnTaking Example TGA Agent")
        group.add_argument(
            "-hid", "--hidden-size", type=int, default=1024, help="Hidden size."
        )
        group.add_argument(
            "--tokenizer_path",
            type=str,
            default="data/turntaking-tokenizer.json",
            help="path to tokenizer.",
        )

    def build_model(self):
        model = ExampleModel(self.dict, self.opt["hidden_size"])
        if self.opt["embedding_type"] != "random":
            self._copy_embeddings(model.embeddings.weight, self.opt["embedding_type"])
        return model

    def eval_step(self, batch):
        # for each row in batch, convert tensor to back to text strings
        return Output([self.dict.vec2txt(row) for row in batch.text_vec])

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return TurnTakingDictionaryAgent


if __name__ == "__main__":
    from argparse import ArgumentParser

    input_string = 'Hello mr. Andersson! Are you? do not say "fine"... :) qwerty'
    opt = {"tokenizer_path": "data/turntaking-tokenizer.json"}
    dict_agent = TurnTakingDictionaryAgent(opt)
    vec = dict_agent.txt2vec(input_string)
    print(vec)
    print(dict_agent.vec2txt(vec))
