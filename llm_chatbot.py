import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Define the model path
model_path = "models/llama-2-7b-chat.Q2_K.gguf"

# Debug prints
print(f"Model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File readable: {os.access(model_path, os.R_OK)}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
try:
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=40,
        n_batch=512,  # Batch size for model processing
        verbose=False,  # Enable detailed logging for debugging
    )
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a RunnableSequence to manage interactions with the prompt and model
llm_chain = prompt | llm

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    # Use invoke instead of run
    answer = llm_chain.invoke({"question": question})
    print(answer, '\n')
