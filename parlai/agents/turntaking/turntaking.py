from parlai.core.torch_agent import History
from parlai.core.torch_agent import Output
from parlai.utils.logging import logging
from parlai.utils.misc import warn_once, recursive_getattr
from parlai.zoo.bert.build import download
import parlai.core.torch_generator_agent as tga

from parlai.agents.transformer.modules import (
    TransformerMemNetModel,
    TransformerGeneratorModel,
    TransformerLinearWrapper,
)

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


def add_common_cmdline_args(argparser):
    """
    Add common command line args.
    """
    argparser.add_argument(
        "-esz",
        "--embedding-size",
        type=int,
        default=300,
        help="Size of all embedding layers",
    )
    argparser.add_argument("-nl", "--n-layers", type=int, default=2)
    argparser.add_argument(
        "-hid",
        "--ffn-size",
        type=int,
        default=300,
        help="Hidden size of the FFN layers",
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout used in Vaswani 2017."
    )
    argparser.add_argument(
        "--attention-dropout",
        type=float,
        default=0.0,
        help="Dropout used after attention softmax.",
    )
    argparser.add_argument(
        "--relu-dropout",
        type=float,
        default=0.0,
        help="Dropout used after ReLU. From tensor2tensor.",
    )
    argparser.add_argument(
        "--n-heads", type=int, default=2, help="Number of multihead attention heads"
    )
    argparser.add_argument("--learn-positional-embeddings", type="bool", default=False)
    argparser.add_argument("--embeddings-scale", type="bool", default=True)
    argparser.add_argument(
        "--n-positions",
        type=int,
        default=None,
        hidden=True,
        help="Number of positional embeddings to learn. Defaults "
        "to truncate or 1024 if not provided.",
    )
    argparser.add_argument(
        "--n-segments",
        type=int,
        default=0,
        help="The number of segments that support the model. "
        "If zero no segment and no langs_embedding.",
    )
    argparser.add_argument(
        "--variant",
        choices={"aiayn", "xlm", "prelayernorm", "bart"},
        default="xlm",
        help="Chooses locations of layer norms, etc. prelayernorm "
        "is used to match some fairseq models",
        recommended="xlm",
    )
    argparser.add_argument(
        "--activation",
        choices={"relu", "gelu"},
        default="relu",
        help="Nonlinear activation to use. AIAYN uses relu, but "
        "more recent papers prefer gelu.",
        recommended="gelu",
    )
    argparser.add_argument(
        "--output-scaling",
        type=float,
        default=1.0,
        help="scale the output of every transformer by this quantity.",
    )
    argparser.add_argument(
        "--share-word-embeddings",
        type="bool",
        default=True,
        help="Share word embeddings table for candidate and context"
        "in the memory network",
    )
    argparser.add_argument(
        "-nel",
        "--n-encoder-layers",
        type=int,
        default=-1,
        help="This will overide the n-layers for asymmetrical transformers",
    )
    argparser.add_argument(
        "-ndl",
        "--n-decoder-layers",
        type=int,
        default=-1,
        help="This will overide the n-layers for asymmetrical transformers",
    )
    argparser.add_argument(
        "--model-parallel",
        type="bool",
        default=False,
        help="Shard the layers across multiple GPUs.",
    )


class TurntakingAgent(tga.TorchGeneratorAgent):
    """
    TurntakingAgent.

    Implementation of TorchGeneratorAgent, where the model is a Transformer
    """

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return TurnTakingDictionaryAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        argparser.set_defaults(
            text_truncate=768,
            label_truncate=256,
            dict_maxexs=0,  # skip building dictionary
        )
        agent = argparser.add_argument_group("Transformer Arguments")
        agent.add_argument(
            "--tokenizer_path",
            type=str,
            default="data/turntaking-tokenizer.json",
            help="path to tokenizer.",
        )

        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)
        super(TurntakingAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt["embedding_type"] != "random":
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt["embedding_type"]
            )
        print(model)
        return model

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when are adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict["embeddings.weight"].size()[0]
        logging.info(f"Resizing token embeddings from {orig_size} to {new_size}")
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            "embeddings.weight",
            "encoder.embeddings.weight",
            "decoder.embeddings.weight",
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict


class TurntakingAgent2(tga.TorchGeneratorAgent):
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
        return Output([str(k) for k in batch.keys()])
        # return Output([self.dict.vec2txt(row) for row in batch.text_vec])

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
