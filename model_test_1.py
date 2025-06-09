"""
Module: model_test_1
This module demonstrates how to send a JSON POST request to a local API endpoint to generate text
using a specified language model. The request is configured to stream the response, making it efficient
for handling large outputs. The module constructs the request payload with details about the model and
input text, sends the request using the requests library, and processes the streaming response by decoding
each received chunk to extract and print the generated text.
Key functionalities:
- Defines a constant URL using typing.Final.
- Constructs a JSON payload with the model name and input text.
- Sends a POST request to the specified URL with a timeout and stream enabled.
- Checks the response status code to ensure successful execution.
- Iterates over the response chunks, decodes them, and prints the generated text.
- Handles errors by printing an appropriate message when the response status code indicates a failure.
Usage:
    Run this module as a standalone script to send a request to the API and print out the generated text.
"""

from typing import Final
import json
import requests

URL: Final = "http://localhost:11434/api/generate"
MODEL: Final = "llama3.2:1b"
PROMPT: Final = "Hello, tell me a short poem about the sea."
TEMPERATURE: Final = 1

data: dict = {
    "model": MODEL,
    "prompt": PROMPT,
    "temperature": TEMPERATURE,
    "think": True,
}

response = requests.post(
    URL,
    json=data,
    stream=True,  # Use stream=True to handle large responses efficiently
    timeout=30,  # Set a timeout for the request
)

# check if the request was successful
if response.status_code == 200:
    print("Response received successfully.")
    print("Generated Response: ", end=" ", flush=True)
    # Process the response in chunks
    for chunk in response.iter_lines():
        if chunk:
            decoded_chunk: str = chunk.decode("utf-8")
            result: dict = json.loads(decoded_chunk)

            generated_text: str = result.get("response", "")
            print(generated_text, end="", flush=True)

else:
    print(f"Request failed with status code: {response.status_code}")
