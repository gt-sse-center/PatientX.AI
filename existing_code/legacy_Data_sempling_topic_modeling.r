# setwd("~/Downloads/2024-04-04")
setwd(file.path(getwd(), "data"))
print(getwd())        # Print the current working directory
print(list.files())   # List files in the current directory

options(repos = c(CRAN = "https://cloud.r-project.org"))

mem.maxVSize(Inf)
mem.maxNSize(Inf)

# brew install gsl
# export LDFLAGS="-L/opt/homebrew/opt/gsl/lib"
# export CPPFLAGS="-I/opt/homebrew/opt/gsl/include"
# install.packages("reshape2")

# install.packages("topicmodels")
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("wordcloud")
# install.packages("RColorBrewer")
# install.packages("syuzhet")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("tidytext")
# install.packages("forcats")

# Loading the libraries

library(topicmodels)
library(tm)

library(SnowballC)
library(wordcloud)

library(RColorBrewer)
library(syuzhet)
library(ggplot2)

library(dplyr)

library(tidytext)

library(forcats)
library(reshape2)


# DATAFILE AND SAMPLING

# >> Loading the sub-forums

data_say_hello <- read.csv("Say hello and introduce yourself.csv")
data_recently_diagnosed <- read.csv("Recently diagnosed and early stages of dementia.csv")
data_memory_concerns <- read.csv("Memory concerns and seeking a diagnosis.csv")
data_i_have_dementia <- read.csv("I have dementia.csv")
data_i_have_partner <- read.csv("I have a partner with dementia.csv")
data_i_care <- read.csv("I care for a person with dementia.csv")


forum_data_union <- rbind(data_say_hello, data_recently_diagnosed, data_memory_concerns, data_i_have_dementia, data_i_have_partner, data_i_care)


# >> Sample from forum_data_union

# sample_size <- 500 
sample_size <- 50 

sample_data <- forum_data_union[sample(nrow(forum_data_union), sample_size),, drop = FALSE]

write.csv(sample_data, "sample_data_final_fixed.csv", row.names = FALSE)


# >> Sample from forum_data_union - first thread posts only

subset_data <- forum_data_union[forum_data_union$message_nr == 1, ]

# sample_size <- 500 
sample_size <- 50 

sample_data_first_thread_post_only <- subset_data[sample(nrow(subset_data), sample_size),, drop = FALSE]

write.csv(sample_data_first_thread_post_only, "sample_data_first_thread_post_only.csv", row.names = FALSE)


# TOPIC MODELING 

# >> data pre-processing

# using unnest_tokens(), removes white spaces, punctuation marks and converts text to lowercase etc.
data_review <- forum_data_union %>%
  unnest_tokens(word, post_message) # What to create (word) from where (reviewText) 
head(data_review)

dim(data_review)

# counting words, count the words and arrange them in descending order to see which words occur more frequently
data_review %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head()

# using unnest_tokens() with stopwords, removes stop words
data_review2 <- forum_data_union %>%
  unnest_tokens(word, post_message) %>%
  anti_join(stop_words)

# After creating intermediate datasets that are no longer needed, you can use rm() to remove them from memory, then call gc() to trigger garbage collection:
rm(forum_data_union)
gc()

# count the words again
data_review2 %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head()

# >> data visualization

# count the words and arrange them in descending order 
print("count the words and arrange them in descending order to see which words occur more frequently")
word_counts <- data_review2 %>%
  count(word) %>%
  filter(n>1000) %>% #number can be changed
  arrange(desc(n)) 

# pass the word_count to gg plot function and flip the axis to see frequency of words
# using coord_flip()
# when data are hard to read
# on the x axis
print("pass the word_count to gg plot function and flip the axis to see frequency of words")
ggplot(word_counts, aes(x=word, y=n)) + 
  geom_col() +
  coord_flip() +
  ggtitle("Forum Word Counts")

# reorder what (word) by what (n)
print("reorder what (word) by what (n)")
word_counts <- data_review2 %>%
  count(word) %>%
  filter(n>1900) %>% #number can be changed
  mutate(word2 = fct_reorder(word, n))

print("word_counts")
word_counts

# now this plot
# with new ordered column x = word2
# is arranged by word count
# and is far better to read:
print("now this plot with new ordered column x = word2 is arranged by word count and is far better to read")
ggplot(word_counts, aes(x=word2, y=n)) + 
  geom_col() +
  coord_flip() +
  ggtitle("Forum Word Counts")

# >> topic modeling

# using as.matrix()
print("Creating Document-Term Matrix (DTM) from the processed data...")
# dtm_review <- data_review2 %>%
#   count(word, word) %>%  # count each word used in each identified review 
#   cast_dtm(word, word, n) %>%  # use the word counts by reviews  to create a DTM
#   as.matrix()

# Step 1: Reduce the vocabulary size
print("Filtering out rare and frequent words to reduce memory usage...")
data_review_filtered <- data_review2 %>%
  count(word) %>%
  filter(n > 2, n < 5000)  # Keep only words that are not too rare or too frequent

# Use the filtered vocabulary to count word occurrences per document
print("Counting the frequency of each word in each document...")
dtm_review <- data_review2 %>%
  filter(word %in% data_review_filtered$word) %>%
  count(document = 1:n(), word) %>%  # Assuming 'document' is a unique identifier for each document
  cast_sparse(document, word, n)  # Use a sparse matrix representation

print("Document-Term Matrix created successfully in sparse format.")

# Proceed with LDA
print("Performing LDA with the reduced and optimized DTM...")
lda_out <- LDA(
  dtm_review,
  k = 2,
  method = "Gibbs",
  control = list(seed = 42)
)

# Inspect the resulting LDA topics
print("LDA model completed. Inspecting the topics...")
lda_topics <- tidy(lda_out, matrix = "beta")
print(head(lda_topics))
# perform LDA,
# k is the number of topics we want to produce,
# specifying the simulation seed helps recover consistent topics
print("perform LDA, k is the number of topics we want to produce, specifying the simulation seed helps recover consistent topics")
lda_out <- LDA(
  dtm_review,
  k = 2,
  method = "Gibbs",
  control = list(seed=42)
)

glimpse(lda_out)

lda_topics <- lda_out %>%
  tidy(matrix = "beta")

lda_topics %>%
  arrange(desc(beta))

#let's do once again topic modeling with LDA()
# and gather all code together: and then
#Finally let's plot discovered topics
# we treat topic as a factor
# to add some color
lda_topics <- LDA(
  dtm_review,
  k = 2,
  method = "Gibbs",
  control = list(seed=42)
) %>%
  tidy(matrix = "beta")
word_probs <- lda_topics %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  mutate(term2 = fct_reorder(term, beta))
ggplot(
  word_probs,
  aes(term2, beta, fill=as.factor(topic))
) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()

# Three topics----
# we will repeat the same steps
# of modeling LDA, tidying, grouping, reordering, and finally ploting
# but with k=3 topics
lda_topics2 <- LDA(
  dtm_review,
  k = 3,
  method = "Gibbs",
  control = list(seed=42)
) %>%
  tidy(matrix = "beta")
word_probs2 <- lda_topics2 %>%
  group_by(topic) %>%
  top_n(15, beta) %>%
  ungroup() %>%
  mutate(term2 = fct_reorder(term, beta))
ggplot(
  word_probs2,
  aes(term2, beta, fill=as.factor(topic))
) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()




