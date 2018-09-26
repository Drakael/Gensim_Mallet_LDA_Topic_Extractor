# Gensim_Mallet_LDA_Topic_Extractor
Python class implementation of Gensim/Mallet topic extration with Latent Dirichlet Allocation from this tutorial https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

Installation :

Run in python console : import nltk; nltk.download('stopwords')

Run in terminal or command prompt : python3 -m spacy download en

Download Mallet from http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
Unzip preferably to C:/Mallet/
Add an environment variable MALLET_HOME with the path to Mallet directory
You might have to update smart_open library

Usage :

gmte = Gensim_Mallet_Topic_Extractor('english')
gmte.extract_topics(data, num_topics) <- data being a list of texts
                                         and num_topics a positive integer

To compute optimal number of topics :

gmte.compute_coherence_values(start=2, limit=40, step=1)

To get most representative documents for each topic :

gmte.compute_coherence_values(start=2, limit=40, step=1)
df_topic_sents_keywords = gmte.format_topics_sentences()
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                             'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

To get topic distribution across documents :

gmte.get_topic_distribution()
