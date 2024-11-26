import requests
import pandas as pd
import json

context_list = []

chat_messages = []

def stream_response(url, payload):
    """Stream responses from a POST request."""
    # print("payload:", payload)
    with requests.post(url, json=payload, stream=False) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        # Parse JSON and extract the 'response' field
                        data = json.loads(line.decode("utf-8"))
                        # print(data)
                        if "context" in data:
                            context_list = data["context"]
                        if "message" in data:
                            print(
                                data["message"]["content"], end=""
                            )  # print when api/generate is used
                            chat_messages.append(data["message"])
                        if "response" in data:
                            print(
                                data["response"], end=""
                            )  # print when api/chat is used
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON: {line}")
        else:
            print(
                f"Failed to retrieve response. Status Code: {response.status_code} Response: {response.text}"
            )


def load_and_preprocess_data(filepath):
    """Load and preprocess CSV data."""
    data = pd.read_csv(filepath)
    # Assuming there's a column named 'text' that contains the posts
    texts = (
        data["post_message"].dropna().tolist()
    )  # Drop missing values and convert to list
    return " ".join(texts)  # Join all texts into a single string


def summarize_text(text, model_url, chunk_size=700):
    """Break text into chunks and send each to the LLM for summarization."""
    # Split the text into chunks of `chunk_size` words
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ][:7]

    # print(chunks[0])
    print(f"\n--- set up context ---\n")
    # use both prompt and messages to be compatible with api/generate and api/chat
    context_prompt = "I am sending you a lot of scraped text from a forum online. Once I am done sending chunks, you will summarize everything I sent. After each chunck tell me you receive it and track how many I sent you."
    chat_messages.append({"role": "user", "content": context_prompt})
    payload = {
        "model": model,
        "prompt": context_prompt,
        "messages": chat_messages,
    }
    stream_response(model_url, payload)

    # Send each chunk to the model
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Sending chunk {i}/{len(chunks)} ---\n")
        prompt_chunk = f"I am sending you a chunk. Here is a chunk: {chunk}"
        chat_messages.append({"role": "user", "content": prompt_chunk})

        payload = {
            "model": model,
            "prompt": prompt_chunk,
            "messages": chat_messages,
            "context": context_list,
        }
        stream_response(model_url, payload)
        print("\n")

    # Signal the model to summarize
    print("\n--- Requesting Final Summary ---\n")
    prompt_final = "I am done sending chunks. Please summarize everything I sent."
    chat_messages.append({"role": "user", "content": prompt_final})
    payload = {
        "model": model,
        "prompt": prompt_final,
        "messages": chat_messages,
        "context": context_list,
    }
    stream_response(model_url, payload)


def print_column_names(filepath):
    """Load the CSV file and print its column names."""
    data = pd.read_csv(filepath)
    print("Column names:", data.columns.tolist())  # Print the column names


# URL to the API
# url = 'http://127.0.0.1:11434/api/generate'
url = "http://127.0.0.1:11434/api/chat"

# Path to the smallest CSV file
filepath = "data/sample_data_first_thread_post_only.csv"

# model to use
model = "mistral-small"
# model = "llama3.2"

# Print the column names to identify the correct column to use
print_column_names(filepath)


# Load and preprocess the data
text = load_and_preprocess_data(filepath)

# Summarize the text
summarize_text(text, url)
