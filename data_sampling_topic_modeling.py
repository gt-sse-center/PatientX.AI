from octis.models.LDA import LDA
import csv
import pandas as pd

# read data
data_say_hello = pd.read_csv('data_say_hello.csv')
data_recently_diagnosed = pd.read_csv('Recently diagnosed and early stages of dementia.csv')
data_memory_concerns = pd.read_csv("Memory concerns and seeking a diagnosis.csv")
data_i_have_dementia = pd.read_csv("I have dementia.csv")
data_i_have_partner = pd.read_csv("I have a partner with dementia.csv")
data_i_care = pd.read_csv("I care for a person with dementia.csv")

# combine data into single dataframe
dfs = [data_say_hello, data_recently_diagnosed, data_memory_concerns, data_i_have_dementia, data_i_have_partner, data_i_care]
forum_data_union = pd.concat(dfs, ignore_index=True)


model = LDA()
