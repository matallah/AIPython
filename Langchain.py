from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import sys
import os
import contextlib

class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        print(token, end='', flush=True)

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

memory = ConversationBufferMemory(memory_key="history", input_key="question")

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.7,
    max_tokens=256,
    verbose=False,
    streaming=True  # Enable streaming
)

template = """
You are a helpful assistant with memory.

History:
{history}

User: {question}
Assistant:
"""

prompt = PromptTemplate(input_variables=["history", "question"], template=template)
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    stream_handler = StreamHandler()

    print("Chatbot: ", end='', flush=True)
    with suppress_stderr():
        response = chain.run(
            question=user_input,
            callbacks=[stream_handler]
        )
    print("\n")