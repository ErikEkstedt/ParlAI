# TurnTaking



## DictionarAgent / Tokenizer

Tokenizer that omits capitalization and punctuation to operate on data more alike spoken-dialog

* Removes: ,.;:?!()[]"
* 10000 tokens


## TransformerGeneratorAgent

```bash
parlai train_model --model turntaking \
  --model-file ./runs/turntaking \
  -t convai2 \
  -bs 128 -eps 20 \
  --tensorboard-log True \
  --tensorboard-logdir ./runs/turntaking
```
