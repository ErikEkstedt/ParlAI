from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import parlai.core.torch_generator_agent as tga

from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip

import torch.nn as nn
import torch.nn.functional as F


##########################################################################
# From ParlAI Example
##########################################################################
class GenModel(tga.TorchGeneratorModel):
    def __init__(self, dictionary, hidden_size=256):
        super().__init__(
            padding_idx=dictionary["padding_idx"],
            start_idx=dictionary["start_idx"],
            end_idx=dictionary["end_idx"],
            unknown_idx=dictionary["unknown_idx"],
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


class SpokenDialogDictionaryAgent(DictionaryAgent):
    """
    Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument("--tokenizer", type=str, default="gpt2")

    @staticmethod
    def build_normalizer():
        normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex('[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
                Strip(),
            ]
        )
        return normalizer

    def __init__(self, opt):
        # initialize from vocab path
        warn_once(
            "WARNING: TurnTaking uses a TurnTaking tokenizer; ParlAI dictionary args are ignored"
        )
        # download(opt["datapath"])
        self._tokenizer = AutoTokenizer.from_pretrained(opt["tokenizer"])
        self.normalizer = SpokenDialogDictionaryAgent.build_normalizer()
        super().__init__(opt)

        if opt["tokenizer"].startswith("gpt2"):
            self.start_token = "[CLS]"
            self.cls_token = "[CLS]"
            self.end_token = "[SEP]"
            self.pad_token = "[PAD]"
            self.null_token = "[PAD]"
            self.mask_token = "[MASK]"
            self.unk_token = "[UNK]"
        elif opt["tokenizer"].startswith("bert"):
            self.start_token = "[bos"
            self.end_token = "[SEP]"
            self.pad_token = "<end-of-text>"
            self.null_token = "<end-of-text>"
            self.mask_token = "[MASK]"
            self.unk_token = "[UNK]"

        self.start_idx = self._tokenizer.convert_tokens_to_ids(self.start_token)
        self.end_idx = self._tokenizer.convert_tokens_to_ids(self.end_token)
        self.pad_idx = self._tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_idx = self._tokenizer.convert_tokens_to_ids(self.unk_token)
        self.null_idx = self._tokenizer.convert_tokens_to_ids(self.null_token)

        self.update_vocab()

    def update_vocab(self):
        # set tok2ind for special tokens
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.pad_token] = self.pad_idx
        self.tok2ind[self.unk_token] = self.unk_idx

        # set ind2tok for special tokens
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.pad_idx] = self.null_token
        self.ind2tok[self.unk_idx] = self.unk_token

    def __len__(self):
        return len(self._tokenizer)

    def encode(self, text, add_special_tokens=False):
        text = self.normalizer.normalize_str(text)
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def txt2vec(self, text, vec_type=list):
        return self.encode(text)

    def vec2txt(self, vec):
        if not isinstance(vec, list):
            # assume tensor
            vec = vec.tolist()
        return self._tokenizer.decode(vec)

    def act(self):
        return {}


class CustomModelAgent(tga.TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.set_defaults(dict_maxexs=0)  # skip building dictionary)
        super(CustomModelAgent, cls).add_cmdline_args(argparser)
        cls.dictionary_class().add_cmdline_args(argparser)
        group = argparser.add_argument_group("Example TGA Agent")
        group.add_argument(
            "-hid", "--hidden-size", type=int, default=256, help="Hidden size."
        )

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return SpokenDialogDictionaryAgent

    def build_model(self):
        model = GenModel(self.dict, self.opt["hidden_size"])
        if self.opt["embedding_type"] != "random":
            self._copy_embeddings(model.embeddings.weight, self.opt["embedding_type"])
        return model


if __name__ == "__main__":
    # Create agent
    args = ["-t", "convai2"]
    parser = ParlaiParser()
    CustomGenAgent.add_cmdline_args(parser)
    parser.add_argument("-n", "--num-examples", default=5, type=int)
    opt = parser.parse_args(args)  # special parlai Opt class, seem to be like a dict

    agent = CustomGenAgent(opt)
    world = create_task(opt, agent)  # DialogPartnerWorld
    for i, agent in enumerate(world.agents):
        print(f"Agent {i}:", agent.__class__.__name__)
        if hasattr(agent, "model"):
            print("model: ", agent.model)

    print("agent.observation: ", agent.observation)
    for i in range(opt["num_examples"]):
        world.parley()
        print(world.display() + "\n")
        if world.epoch_done():
            print("epoch done")
            break
    print("agent.observation:\n", agent.observation["text_vec"])
