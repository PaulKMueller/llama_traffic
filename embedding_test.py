import torch
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Encode the input text
input_text = "Hello, BE!"
encoded_input = tokenizer(input_text, return_tensors="pt")

# Get BERT embeddings
with torch.no_grad():
    output = model(**encoded_input)
    # `output` is a tuple, with the first element being the sequence of hidden-states at the output of the last layer
    # It has shape (batch_size, sequence_length, hidden_size), so we pick the first item in the sequence (for [CLS]) as our embedding for the whole sentence
    embedding = output[0][0][0]

print(embedding)
