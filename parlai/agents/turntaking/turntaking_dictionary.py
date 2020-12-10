from parlai.core.dict import DictionaryAgent
from parlai.zoo.bert.build import download
from parlai.utils.misc import warn_once

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers, normalizers, decoders, Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece

from os.path import join

try:
    from tokenizers import Tokenizer
except ImportError:
    raise ImportError(
        "TurnTaking requires Huggingface tokenizers installed. \n"
        "pip install tokenizers"
    )


class TurnTakingTokenizer(object):
    @staticmethod
    def train(
        tokenizer, files, vocab_size, directory="data", name="turntaking-tokenizer"
    ):
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        )
        tokenizer.train(trainer, files)
        model_files = tokenizer.model.save(directory, name)
        savepath = join(directory, name + ".json")
        tokenizer.save(savepath, pretty=True)
        return tokenizer, savepath, model_files

    @staticmethod
    def build_tokenizer():
        tokenizer = Tokenizer(WordPiece())
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Punctuation(), Whitespace()])
        tokenizer.normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex('[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),
                Strip(),
            ]
        )
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer


class TurnTakingDictionaryAgent(DictionaryAgent):
    """
    Allow to use the Torch Agent with the wordpiece dictionary of Hugging Face.
    """

    def __init__(self, opt):
        super().__init__(opt)
        # initialize from vocab path
        warn_once(
            "WARNING: TurnTaking uses a TurnTaking tokenizer; ParlAI dictionary args are ignored"
        )
        # download(opt["datapath"])
        self.tokenizer = Tokenizer.from_file(opt["tokenizer_path"])

        self.start_token = "[CLS]"
        self.end_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.null_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"

        self.start_idx = self.tokenizer.token_to_id(self.start_token)
        self.end_idx = self.tokenizer.token_to_id(self.end_token)
        self.pad_idx = self.tokenizer.token_to_id(self.pad_token)
        self.unk_idx = self.tokenizer.token_to_id(self.unk_token)
        self.null_idx = self.tokenizer.token_to_id(self.null_token)

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

    def txt2vec(self, text, vec_type=list):
        tokens = self.tokenizer.encode(text)
        return tokens.ids

    def vec2txt(self, vec):
        if not isinstance(vec, list):
            # assume tensor
            vec = vec.tolist()
        return self.tokenizer.decode(vec)

    def act(self):
        return {}


# TODO:
# Should only be done once to save a tokenizer
def get_files():
    from glob import glob

    files = []
    # files = [
    #     f"data/wikitext-103-raw/wiki.{split}.raw"
    #     for split in ["test", "train", "valid"]
    # ]
    files += glob(join("data/ConvAI2", "*both*.txt"))
    files += glob(join("data/CornellMovie", "*.txt"))
    files += glob(join("data/Twitter", "*.txt"))
    return files


if __name__ == "__main__":

    input_string = 'Hello mr. Andersson! Are you? do not say "fine"... :) qwerty'
    opt = {"tokenizer_path": "data/turntaking-tokenizer.json"}
    dict_agent = TurnTakingDictionaryAgent(opt)
    vec = dict_agent.txt2vec(input_string)
    print(vec)
    print(dict_agent.vec2txt(vec))
