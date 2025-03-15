from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with the correct model identifier (if available)
model_name = "intfloat/multilingual-e5-large"  # example identifier

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

def get_embedding(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the last hidden state
    last_hidden = outputs.hidden_states[-1]  # shape: [batch_size, seq_length, hidden_dim]
    # Mean pooling to get fixed size vector
    embedding = last_hidden.mean(dim=1).squeeze()
    return embedding.cpu().numpy()

arabic_text = "هذه جملة عربية لاستخراج التمثيل الرقمي"
embedding_vector = get_embedding(arabic_text)
print("Embedding:", embedding_vector)
