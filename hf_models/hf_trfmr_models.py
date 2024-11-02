## SOURCE -- https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens


from util_logger import setup_logger
logger = setup_logger(module_name=str(__name__))

import streamlit as st

class HFModelsInvoke:
    """
    """
    def __new__(cls):
        if not hasattr(cls,'instance'):
            cls.instance = super(HFModelsInvoke,cls).__new__(cls)

        return cls.instance

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="HF1BitLLM/Llama3-8B-1.58-100B-tokens")
pipe(messages)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens")
model = AutoModelForCausalLM.from_pretrained("HF1BitLLM/Llama3-8B-1.58-100B-tokens")
