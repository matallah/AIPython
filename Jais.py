from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_embedding(text):
    """
    Generates an embedding for the input text by:
    - Tokenizing the input.
    - Running the model to extract hidden states.
    - Applying mean pooling on the last hidden state.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run the model and obtain hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the last hidden state and perform mean pooling along the sequence dimension
    last_hidden_state = outputs.hidden_states[-1]  # shape: [batch_size, sequence_length, hidden_size]
    embedding = last_hidden_state.mean(dim=1).squeeze()  # shape: [hidden_size]

    return embedding.numpy()

# Replace 'your_huggingface_token' with your actual Hugging Face token if needed.
# Ensure you have access to the gated model "inceptionai/jais-13b".
tokenizer = AutoTokenizer.from_pretrained("inceptionai/jais-13b", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained("inceptionai/jais-13b", output_hidden_states=True, use_auth_token=True)

# Example Arabic text for which we want to obtain an embedding
arabic_text = "هذه جملة باللغة العربية لاستخراج التمثيل الرقمي"
embedding = get_embedding(arabic_text)

print("Embedding vector:", embedding)
