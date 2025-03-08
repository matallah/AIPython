from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory  # New import for memory
import sys
import os
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

model_path = "/Users/mo/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"

# Initialize memory
memory = ConversationBufferMemory(memory_key="history", input_key="question")

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=256,
    verbose=False
)

# Updated template with history
template = """
You are a helpful assistant with memory.

History:
{history}

User: {question}
Assistant:
"""

prompt = PromptTemplate(input_variables=["history", "question"], template=template)

# Add memory to the chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    with suppress_stderr():
        response = chain.run(question=user_input)

    print(f"Chatbot: {response}\n")