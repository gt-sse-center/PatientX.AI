setwd("~/Downloads/2024-04-04")

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

sample_size <- 500 

sample_data <- forum_data_union[sample(nrow(forum_data_union), sample_size),, drop = FALSE]

write.csv(sample_data, "sample_data_final_fixed.csv", row.names = FALSE)


# >> Sample from forum_data_union - first thread posts only

subset_data <- forum_data_union[forum_data_union$message_nr == 1, ]

sample_size <- 500 

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

# count the words again
data_review2 %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head()

# >> data visualization

# count the words and arrange them in descending order 
word_counts <- data_review2 %>%
  count(word) %>%
  filter(n>1000) %>% #number can be changed
  arrange(desc(n)) 

# pass the word_count to gg plot function and flip the axis to see frequency of words
# using coord_flip()
# when data are hard to read
# on the x axis
ggplot(word_counts, aes(x=word, y=n)) + 
  geom_col() +
  coord_flip() +
  ggtitle("Forum Word Counts")

# reorder what (word) by what (n)
word_counts <- data_review2 %>%
  count(word) %>%
  filter(n>1900) %>% #number can be changed
  mutate(word2 = fct_reorder(word, n))

word_counts

# now this plot
# with new ordered column x = word2
# is arranged by word count
# and is far better to read:
ggplot(word_counts, aes(x=word2, y=n)) + 
  geom_col() +
  coord_flip() +
  ggtitle("Forum Word Counts")

# >> topic modeling

# using as.matrix()
dtm_review <- data_review2 %>%
  count(word, word) %>%  # count each word used in each identified review 
  cast_dtm(word, word, n) %>%  # use the word counts by reviews  to create a DTM
  as.matrix()

# perform LDA,
# k is the number of topics we want to produce,
# specifying the simulation seed helps recover consistent topics
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




