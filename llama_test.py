from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer


def get_llama_embedding(input_text):
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    input_ids = tokenizer.encode(input_text, return_tensors="np")
    return input_ids


# Generate text
# The `generate` method returns token ids which we can decode back to text
# output_ids = model.generate(input_ids, max_length=100)  # you can customize max_length
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print(output_text)
