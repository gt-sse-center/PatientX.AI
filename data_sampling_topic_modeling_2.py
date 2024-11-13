import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Set working directory to 'data' folder
# This line sets the working directory for all file operations
os.chdir(os.path.join(os.getcwd(), "data"))
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir()}\n")

# Load CSV files into DataFrames
# Each data frame represents data from specific sub-forums
data_say_hello = pd.read_csv("Say hello and introduce yourself.csv")
data_recently_diagnosed = pd.read_csv("Recently diagnosed and early stages of dementia.csv")
data_memory_concerns = pd.read_csv("Memory concerns and seeking a diagnosis.csv")
data_i_have_dementia = pd.read_csv("I have dementia.csv")
data_i_have_partner = pd.read_csv("I have a partner with dementia.csv")
data_i_care = pd.read_csv("I care for a person with dementia.csv")

# Combine all data into one DataFrame
forum_data_union = pd.concat([data_say_hello, data_recently_diagnosed, data_memory_concerns,
                              data_i_have_dementia, data_i_have_partner, data_i_care], ignore_index=True)

# Sampling from the combined data
# Sampling is done to reduce the data size and improve processing time
sample_size = 100
sample_data = forum_data_union.sample(n=sample_size, random_state=42)
sample_data.to_csv("sample_data_final_fixed.csv", index=False)
print("Sample data saved as 'sample_data_final_fixed.csv'\n")

# Sampling only the first thread posts
# Filtering and sampling the first message from each thread
subset_data = forum_data_union[forum_data_union['message_nr'] == 1]
sample_data_first_thread_post_only = subset_data.sample(n=sample_size, random_state=42)
sample_data_first_thread_post_only.to_csv("sample_data_first_thread_post_only.csv", index=False)
print("Sample of first thread posts saved as 'sample_data_first_thread_post_only.csv'\n")

# Text Pre-processing
print("Pre-processing the text data...")
# Combining all post messages into one column
forum_data_union['combined_text'] = forum_data_union.apply(lambda row: ' '.join(str(val) for val in row if isinstance(val, str)), axis=1)

# Removing punctuation, converting to lowercase, and tokenizing
forum_data_union['processed_text'] = forum_data_union['combined_text'].apply(lambda x: re.sub(r'\W+', ' ', x.lower()))
print(f"Pre-processed data sample: {forum_data_union['processed_text'].head()}\n")

# Removing stopwords
print("Removing stopwords...")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
forum_data_union['processed_text'] = forum_data_union['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS]))

# Removing all numbers
print("Removing all numbers from the text...")
forum_data_union['processed_text'] = forum_data_union['processed_text'].apply(lambda x: re.sub(r'\d+', '', x))
print(f"Data after removing numbers: {forum_data_union['processed_text'].head()}\n")





# Filtering out rare and frequent words to reduce memory usage
print("Filtering out rare and frequent words...")
vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
X = vectorizer.fit_transform(forum_data_union['processed_text'][:20000])
print("Vectorizer vocabulary size:", len(vectorizer.vocabulary_))
# Printing a small portion of the Document-Term Matrix to see its contents
print("Inspecting part of the Document-Term Matrix (sparse matrix)...")
print(X[:10, :10].toarray())  # Print a small 10x10 portion of the matrix to see what it looks like



# Creating the Document-Term Matrix (DTM)
print("Creating Document-Term Matrix (DTM) from the processed data...")
# Sparse matrix representation is used to save memory
dtm = X.toarray()
print(f"Document-Term Matrix shape: {dtm.shape}\n")

# Perform LDA Topic Modeling
print("Performing LDA Topic Modeling with 2 topics...")
lda = LDA(n_components=8, random_state=42)
lda.fit(dtm)

# Inspecting the topics
print("LDA model completed. Inspecting the topics...")
topics = lda.components_
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(topics):
    print(f"Topic {idx}:\n{' '.join([feature_names[i] for i in topic.argsort()[:-21:-1]])}\n")

# Visualizing topics using WordCloud
print("Generating word clouds for each topic...")
for idx, topic in enumerate(topics):
    word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[:-21:-1]}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic {idx}")
plt.show()

# Performing LDA with 3 topics and visualizing
print("Performing LDA with 3 topics...")
lda3 = LDA(n_components=3, random_state=42)
lda3.fit(dtm)

# Inspecting topics from the new LDA model
print("LDA model with 3 topics completed. Inspecting the topics...")
topics3 = lda3.components_
for idx, topic in enumerate(topics3):
    print(f"Topic {idx}:\n{' '.join([feature_names[i] for i in topic.argsort()[:-21:-1]])}\n")

print("Generating word clouds for each topic from the 3-topic model...")
for idx, topic in enumerate(topics3):
    word_freq = {feature_names[i]: topic[i] for i in topic.argsort()[:-21:-1]}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic {idx} - Model with 3 Topics")
plt.show()
