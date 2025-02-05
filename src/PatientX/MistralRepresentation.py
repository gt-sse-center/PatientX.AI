import requests
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document

DEFAULT_PROMPT = """
I have topic that contains the following documents: \n[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the above information, can you give a short label of the topic?
"""


DEFAULT_PROMPT_CHAT_START = """
I will send you documents. After I send you the documents, I will ask you to write a short description capturing the commonalities across all documents.
"""

DEFAULT_PROMPT_CHAT_CONTEXT = """
This is one of the documents:
[DOCUMENTS]
"""

DEFAULT_PROMPT_CHAT_ENDING = """
I need you to write "The topic is:" then print a short description of the documents in markdown format.
"""

class MistralRepresentation(BaseRepresentation):
    def __init__(
        self,
        model: str = "mistral-small",
        prompt: str = None,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float = None,
        exponential_backoff: bool = False,
        chat: bool = False,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        api: str = "generate",
    ):
        self.model = model
        self.api = api
        if prompt is None:
            self.prompt = DEFAULT_PROMPT if chat else DEFAULT_PROMPT
        else:
            self.prompt = prompt

        self.default_prompt_ = DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.chat = True if self.api == "chat" else False
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        self.prompts_ = []
        self.chat_messages = []
        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
            del self.generator_kwargs["model"]
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop") and not chat:
            self.generator_kwargs["stop"] = "\n"
        
    def stream_response(self, url, payload) -> str:
        """
        Stream responses from a POST request to a specified URL.

        Arguments:
            url (str): The endpoint URL to which the POST request will be sent.
            payload (dict): A dictionary containing the data to be sent in the POST request. 
                            Typically includes the model and input data required for processing.

        Returns:
            str: The concatenated response content from the server. This could include 
                either 'message' content (for api/generate) or 'response' content 
                (for api/chat), depending on the API configuration.
        """
        response_text = ""
        with requests.post(url, json=payload, stream=False) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse JSON and extract the 'response' field
                            data = json.loads(line.decode("utf-8"))
                            if "message" in data:
                                print(
                                    data["message"]["content"], end=""
                                )  # print when api/generate is used

                                response_text += data["message"]["content"]

                                self.chat_messages.append(data["message"])
                            if "response" in data:
                                print(
                                    data["response"], end=""
                                )  # print when api/chat is used

                                response_text += data["response"]

                        except json.JSONDecodeError:
                            print(f"Failed to decode JSON: {line}")
            else:
                print(
                    f"Failed to retrieve response. Status Code: {response.status_code} Response: {response.text}"
                )
        return response_text

    def get_response(self, model, model_url, prompt, messages, generate):
        payload = {
            "model": model,
            "prompt": prompt,
            "messages": messages,
        }

        if generate:
            payload = {
                "model": model,
                "prompt": prompt,
            }

        return stream_response(model_url, payload)

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        model = "mistral-small"
        url = "http://127.0.0.1:11434/api/" + self.api

        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using Mistral's Language Model
        updated_topics = {}
        if self.api == "chat":
            self.chat_messages = []
            self.chat_messages.append({"role": "user", "content": DEFAULT_PROMPT_CHAT_START})
            payload = {
                "model": model,
                "messages": self.chat_messages,
            }
            response = self.stream_response(url, payload)
        
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            response = ""
            if self.api == "chat":
                context_prompt = DEFAULT_PROMPT_CHAT_CONTEXT
                # make loop for each truncated_docs so feed one prompt at a time
                for doc in truncated_docs:
                    context_prompt = self._replace_documents(context_prompt, [doc])
                    self.chat_messages.append({"role": "user", "content": context_prompt})
                    payload = {
                        "model": model,
                        "messages": self.chat_messages,
                    }
                    response = self.stream_response(url, payload)
                    
                # now ask the topic to llm
                self.chat_messages.append({"role": "user", "content": DEFAULT_PROMPT_CHAT_ENDING})
                payload = {
                    "model": model,
                    "messages": self.chat_messages,
                }
                response = self.stream_response(url, payload)

            if self.api == "generate":
                response = self.get_response(model, url, prompt, messages=prompt, generate=True)

            # Extract the topic name from the response
            topic_name = self._extract_topic_name(response)
            updated_topics[topic] = [(topic_name, 1)]
            # updated_topics[topic] = [(label, 1)]

        return updated_topics
    
    def _extract_topic_name(self, response: str) -> str:
        """Extract the topic name from the response.

        Arguments:
            response: The response from the Mistral model

        Returns:
            topic_name: The extracted topic name
        """
        # Assuming the response format is consistent and the topic name is prefixed with "Topic: "
        topic_prefix = "Topic: "
        if topic_prefix in response:
            topic_name = response.split(topic_prefix)[1].strip()
            return topic_name
        else:
            # Fallback to returning the entire response if the prefix is not found
            return response.strip()

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        prompt = self.prompt
        if "[KEYWORDS]" in prompt:
            prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
        if "[DOCUMENTS]" in prompt:
            prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt