
# Moving Away from LDA: Adopting BERTopic and LLM-based Clustering

## Overview

After thorough experimentation with Latent Dirichlet Allocation (LDA) for extracting topics from patient forum data related to dementia, we have decided to shift away from this approach. LDA has provided some initial insights, but it has consistently fallen short of capturing the deeper, nuanced themes expressed in patient experiences. This limitation became particularly evident in the context of highly complex, emotionally rich forum posts that reflect the lived experiences of patients and caregivers dealing with dementia.

Instead, we are now focusing on two more advanced approaches: **BERTopic** and **cluster generation using Large Language Models (LLMs)**. These methods offer more sophisticated ways to understand and represent the intricate patterns and latent themes within our data.

## Challenges with LDA

LDA is a popular topic modeling technique that works well for many applications. However, for our use case of analyzing patient experiences through forum posts, several limitations became evident:

- **Lack of Contextual Understanding**: LDA generates topics by finding co-occurrences of words without understanding their context. Many patient posts involve complex emotions, medical terminologies, and contextual subtleties that LDA fails to interpret effectively.

- **Limited Coherence**: The topics produced by LDA often lack coherence and do not align well with human-understandable concepts. Many of the terms grouped into topics by LDA appeared unrelated or too generic to be useful for driving actionable insights.

- **Rigid Bag-of-Words Model**: The traditional bag-of-words model used by LDA does not capture word relationships, meaning the insights generated were simplistic and often lacked depth.

- **Need for Heavy Pre-Processing**: Forum threads often contain slang, abbreviations, typos, and informal grammar, making traditional pre-processing (e.g., stemming, lemmatization) less effective and prone to errors. LDA struggles in this context because its bag-of-words approach doesn’t account for the nuances of such informal language, leading to less coherent topics. In contrast, LLMs and BERTopic handle colloquial data more effectively by leveraging contextual embeddings that understand variations in language use and capture semantic meaning, even in noisy or informal text.

## Transition to BERTopic and LLM-based Clustering

To address the challenges faced with LDA, we are now adopting the following approaches:

### 1. BERTopic

BERTopic is an advanced topic modeling method that leverages transformer-based embeddings to create better topic representations. The key advantages include:

- **Contextual Embeddings**: BERTopic uses embeddings from transformer models, such as BERT, to understand the context and relationships between words, allowing it to produce more meaningful topics.

- **Dynamic Clustering**: It incorporates density-based clustering techniques to dynamically generate topics, leading to more natural groupings that align better with patient experiences.

- **Better Interpretability**: The generated topics are typically more interpretable and closer to the kind of high-level insights we are aiming to provide to healthcare professionals and patients.

### 2. Clustering with Large Language Models (LLMs)

In addition to BERTopic, we are exploring the use of LLMs, such as LLaMA, to directly assist in generating topic clusters. LLMs offer the following benefits:

- **Complex Insight Generation**: LLMs can comprehend and summarize complex and nuanced content, which allows us to generate more insightful and human-readable topic clusters from the patient posts.

- **Flexibility**: Using LLMs, we can apply prompts that adapt to the specific type of insights we are looking for, which is particularly helpful in understanding the lived experiences of patients along their healthcare journey.

- **Human-like Interpretation**: LLMs are capable of simulating human-like understanding, making them highly suitable for interpreting the emotional and psychological aspects of patient narratives.

## Next Steps

- We will be training and finer-tuning BERTopic on our existing forum dataset to extract more meaningful and contextual topics.
- We will also integrate try LLMs into our pipeline to help generate insightful clusters that resonate with the real-life experiences of patients and caregivers.

This shift in methodology reflects our commitment to delivering a deeper and more accurate representation of patient experiences, ultimately helping improve healthcare processes for individuals with dementia.

If you have questions or suggestions about this new approach, feel free to reach out. We believe this new direction will significantly improve the quality and utility of the insights generated.

