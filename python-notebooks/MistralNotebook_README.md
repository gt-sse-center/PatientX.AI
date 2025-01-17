# Documentation for LLM Integration in PatientX.AI

This document explains how to utilize the LLM-related components in the **PatientX.AI** project, specifically focusing on:

1. **Using BERTopic with LLMs**
2. **Understanding the `MistralRepresentation` Class and Customizing Prompts**
4. **Interacting with Chat/Generate APIs**

---

## 1. Using BERTopic with LLMs

The project integrates BERTopic to perform topic modeling. Here's a brief overview of the setup in the `bertopic.ipynb` notebook:

### Prerequisites

Ensure you have the required libraries installed:
```bash
# install the following packages, depending on your system, you could use regular pip
pip install --upgrade pip
pip3 install numpy==1.24.4
pip3 install bertopic
pip3 install spacy
pip3 install datamapplot
pip3 install "nbformat>=4.2.0"
pip3 install --upgrade nbformat
pip3 install ipykernel

```
---

## Advanced Usage of `MistralRepresentation`

This section provides deeper insights into using the `MistralRepresentation` class for advanced tasks.

### 1. Streaming Responses from APIs

The `stream_response` method allows you to handle responses incrementally when working with APIs that support streaming. This is particularly useful when generating lengthy responses.

#### Example:
```python
url = "http://127.0.0.1:11434/api/generate"
payload = {
    "model": "mistral-small",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "prompt": "What is the capital of France?"
}
#using "messages" and ""prompt"" is compatible with api/generate and api/chat

response = mistral_representation.stream_response(url, payload)
print(response)
```

## 2. Customizing Prompts

Prompts are central to how the LLM interprets the input.

#### Default Prompts

The default prompts are defined as constants in the `MistralRepresentation` class. You can modify the prompts as needed for your specific use case.:

	•	DEFAULT_PROMPT:
```python
Here are documents:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]
I need you to write "The topic is:" then print a short description of the documents in markdown format.
```

Using the chat API, the default prompt composed of the following elements:

    •	DEFAULT_PROMPT_CHAT_START:
    •	DEFAULT_PROMPT_CHAT_CONTEXT:
    •	DEFAULT_PROMPT_CHAT_END:

Start will be called first and then context for each document and end will be called to ask for the topic.

---

## Conclusion

The integration of `MistralRepresentation` within the PatientX.AI project provides a robust framework for leveraging LLMs in topic modeling and text analysis. By utilizing customizable prompts, flexible API configurations, and advanced handling of documents, this implementation allows for dynamic and accurate representations of text-based data. 

Key takeaways from this documentation include:
- Setting up and using BERTopic for initial topic modeling.
- Understanding and extending the functionality of the `MistralRepresentation` class.
- Interacting with the Chat/Context and Generate APIs for diverse use cases.
- Managing prompts and fine-tuning parameters to optimize results.

This setup enables scalable and efficient processing of large text datasets while offering flexibility to adapt to evolving requirements. Future updates and enhancements will continue to refine the framework, ensuring it remains a powerful tool for text-based AI applications.


---