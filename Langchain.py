from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import sys
import os
import contextlib

# Helper context manager to suppress stderr
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# Path to your local GGUF model file.
model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"

# Initialize the LlamaCpp LLM with your model file.
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=256,
    verbose=False  # Attempt to disable internal logging
)

# Define a prompt template for the chatbot.
template = (
    "You are a helpful assistant.\n\n"
    "User: {question}\n"
    "Assistant:"
)
prompt = PromptTemplate(input_variables=["question"], template=template)

# Create an LLMChain that connects the prompt with the model.
chain = LLMChain(llm=llm, prompt=prompt)

# Start the chatbot loop.
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]\
            :
        break

    # Option 1: Suppress extra logs during generation.
    with suppress_stderr():
        response = chain.run(question=user_input)

    # Option 2: If you prefer not using stderr suppression, simply uncomment:
    # response = chain.run(question=user_input)

    print(f"Chatbot: {response}\n")
