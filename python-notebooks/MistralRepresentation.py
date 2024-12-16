from interact_with_mistral import *
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document


DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the worst food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_CHAT_PROMPT = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
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
    ):
        self.model = model

        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        else:
            self.prompt = prompt

        self.default_prompt_ = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.chat = chat
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer

        self.prompts_ = []

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
            del self.generator_kwargs["model"]
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop") and not chat:
            self.generator_kwargs["stop"] = "\n"

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
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using Mistral's Language Model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            self.prompts_.append(prompt)

            model = "mistral-small"
            url = "http://127.0.0.1:11434/api/generate"

            label = get_response(model, url, prompt, messages=prompt, generate=True)
            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT or self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
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