import torch
from transformers import T5Tokenizer, T5EncoderModel

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

model = T5EncoderModel.from_pretrained(DEFAULT_T5_NAME)
tokenizer = T5Tokenizer.from_pretrained(DEFAULT_T5_NAME)

sentence ="In the year 2525 if man is still alive, if woman can survive they will find"

tokens = tokenizer.encode(sentence)

model.eval()
output = model(tokens)

torch.onnx.export(
    model, 
    (torch.randint(0, 8000, (1,256,)), torch.randint(0, 2, (1,256,))), 
    '../transformers-js/models/t5-model.onnx', 
    input_names=['tokens', 'attention_mask'], 
    output_names=['encoding'],
    dynamic_axes={'tokens': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}, 'encoding': {0: 'batch_size', 1: 'sequence_length'}}
)

print("export complete")