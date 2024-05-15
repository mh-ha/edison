import os
import sys

sys.path.append(os.path.abspath('/home/maengmaengeeee/project/DeBERTa'))
print(sys.path)


from DeBERTa import deberta

print(deberta)
model = deberta.DeBERTa(pre_trained='base')
vocab_path, vocab_type = model.load_vocab(pretrained_id='base')
tokenizer = model.tokenizers[vocab_type](vocab_path)
print(vocab_path, vocab_type, tokenizer)
print(model)