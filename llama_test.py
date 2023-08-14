from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')

# Text to generate from
input_text = "Produce a set of Vectors in the following format: \n V: 1, 3, 6, 2, 5"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
# The `generate` method returns token ids which we can decode back to text
output_ids = model.generate(input_ids, max_length=50)  # you can customize max_length
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(output_text)

