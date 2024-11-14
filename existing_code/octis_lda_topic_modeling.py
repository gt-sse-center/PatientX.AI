from octis.models.LDA import LDA
import pandas as pd
import string
from octis.preprocessing.preprocessing import Preprocessing
import os

print("reading data...")

data_folder_path = os.path.join(os.getcwd(), "forum-crawler-data")

# read data
data_say_hello = pd.read_csv(os.path.join(data_folder_path, 'Say hello and introduce yourself.csv'))
data_recently_diagnosed = pd.read_csv(os.path.join(data_folder_path, 'Recently diagnosed and early stages of dementia.csv'))
data_memory_concerns = pd.read_csv(os.path.join(data_folder_path, 'Memory concerns and seeking a diagnosis.csv'))
data_i_have_dementia = pd.read_csv(os.path.join(data_folder_path, 'I have dementia.csv'))
data_i_have_partner = pd.read_csv(os.path.join(data_folder_path, 'I have a partner with dementia.csv'))
data_i_care = pd.read_csv(os.path.join(data_folder_path, 'I care for a person with dementia.csv'))


print("read data")


# combine data into single dataframe
dfs = [data_say_hello, data_recently_diagnosed, data_memory_concerns, data_i_have_dementia, data_i_have_partner, data_i_care]
forum_data_union = pd.concat(dfs, ignore_index=True)


SAMPLE_SIZE = 50

sample_data = forum_data_union.sample(SAMPLE_SIZE)

print("sampled")

# Save to CSV file

# save to TSV file, train, test, val splits
sample_data.to_csv(path_or_buf='/Users/vnarayan35/Documents/GitHub/PatientX.AI/existing_code/dataset/sample_data_final_fixed.tsv', index=False, sep='\t')

# data that we will use for LDA
data_review = forum_data_union['post_message'] # hold only text from posts
data_review.to_csv(r'./corpus.txt', header=None, index=None, sep='\n', mode='a')

print("preprocessing...")

# preprocessing - remove whitespace, remove punctuation, convert to lowercase
preprocessor = Preprocessing(vocabulary=None, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=1, min_words_docs=0)

dataset = preprocessor.preprocess_dataset(documents_path=r'./corpus.txt')
print("done preprocessing")

print("saving...")
dataset.save(path='./processed_dataset/')
print("done saving")



# Topic modeling

model = LDA(num_topics=2, alpha=0.1)


print("training lda...")

# Train the model using default partitioning choice
output = model.train_model(dataset)

print("done training")

print(*list(output.keys()), sep="\n") # Print the output identifiers

for t in output['topics'][:5]:
  print(" ".join(t))


