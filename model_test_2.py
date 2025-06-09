"""
Module: model_test_2

This module demonstrates the use of the Ollama API to interact with language models.
It includes the following operations:
1. Listing available models using olm.list().
2. Chatting with a specific model (MODEL) via olm.chat() with streaming of the response.
3. Generating text from a prompt using olm.generate().
4. Displaying information about the selected model with olm.show().
5. Creating a custom model ('astronomy_expert') based on an existing one by applying a system instruction and parameter settings.
6. Generating a response using the newly created custom model.
7. Removing the custom model via olm.delete() to clean up.

The module leverages constants for fixed parameters and demonstrates a typical workflow for testing and manipulating models using the Ollama API.
"""

from typing import Final, Any, Dict, List
import pprint as pp
import ollama as olm

# Define constant variables
MODEL: Final[str] = "llama3.2:1b"
TEMPERATURE: Final[float] = 0.1
PROMPT_MATH: Final[str] = "Hello, tell me a short poem about the mathematics."
PROMPT_UNIVERSE: Final[str] = "Hello, tell me a short poem about the universe."
CUSTOM_MODEL_NAME: Final[str] = "astronomy_expert"
CUSTOM_MODEL_SYSTEM: Final[str] = (
    "You are an erudite assistant with a profound mastery of the astronomy's secrets. "
    "Your responses are concise yet filled with illuminating insights."
)
CUSTOM_MODEL_PARAMETERS: Final[Dict[str, float]] = {"temperature": TEMPERATURE}

# List available models
models_lists: Any = olm.list()
print("Available models: \n")
pp.pprint(models_lists)  # Pretty print the list

# Chat with the model using Ollama's chat API
res = olm.chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": PROMPT_UNIVERSE},
    ],
    stream=True,
)

print("Poem about the universe via chat: \n")
for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)

# Generate example text using Ollama's generate API
res = olm.generate(
    model=MODEL,
    prompt=PROMPT_MATH,
)
print(f"\n\nPoem about mathematics via generate: \n{res['response']}")

# See the model details
model_details: Any = olm.show(MODEL)
print(model_details)

# Create a custom model using Ollama's create API
olm.create(
    model=CUSTOM_MODEL_NAME,
    from_=MODEL,
    system=CUSTOM_MODEL_SYSTEM,
    parameters=CUSTOM_MODEL_PARAMETERS,
)

# Use the newly created model to generate a response
print(f"\n\nUsing the custom model '{CUSTOM_MODEL_NAME}' to answer a question:\n")
res = olm.generate(
    model=CUSTOM_MODEL_NAME,
    prompt="What is the distance from Earth to the nearest star other than the Sun?",
)

print(f"Response from the custom model '{CUSTOM_MODEL_NAME}': \n")
print(res["response"])

# Delete the model as we don't want it to be persistent
print(f"\n\nDeleting the custom model '{CUSTOM_MODEL_NAME}'...\n")
olm.delete(CUSTOM_MODEL_NAME)
print(f"Custom model '{CUSTOM_MODEL_NAME}' deleted successfully.")
