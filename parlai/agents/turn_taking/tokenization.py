from glob import glob
from os.path import join

from tokenizers.models import WordPiece
from tokenizers import pre_tokenizers, normalizers, decoders, Regex
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Replace, Strip
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


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


# TODO:
# Should only be done once to save a tokenizer
def get_files():
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

    files = get_files()

    tokenizer = TurnTakingTokenizer.build_tokenizer()
    tokenizer, savepath, model_files = TurnTakingTokenizer.train(
        tokenizer, files, vocab_size=10000
    )

    input_string = 'Hello mr. Andersson! Are you? do not say "fine"... :) qwerty'
    output = tokenizer.encode(input_string)
    print(output.ids)
    print(output.tokens)
    dec = tokenizer.decode(output.ids)
    print(dec)
    # print(dec)
