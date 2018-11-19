import re
import sys
import pandas as pd
import numpy as np
from pprint import pprint
# from Gensim_Mallet_Topic_Extractor import GensimMalletTopicExtractor
from msai.Gensim_Mallet_Topic_Extractor import GensimMalletTopicExtractor

if __name__ == '__main__':
    # freeze_support()

    gmte = GensimMalletTopicExtractor('english')

    # Import Dataset
    df = pd.read_json('newsgroups.json')
    print(df.target_names.unique(), '\n')

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove Nntp-Posting-Host
    data = [sent.replace('From:', '') for sent in data]
    data = [sent.replace('Organization:', '') for sent in data]
    data = [sent.replace('Subject:', '') for sent in data]
    data = [sent.replace('Summary:', '') for sent in data]
    data = [sent.replace('Keywords:', '') for sent in data]
    data = [sent.replace('Distribution:', '') for sent in data]
    data = [sent.replace('Lines:', '') for sent in data]
    data = [sent.replace('Nntp-Posting-Host:', '') for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    pprint(data[5])
    np.random.seed(1)
    np.random.shuffle(data)
    gmte.extract_topics(data[:150], 20,
                        # passes=10,
                        iterations=500,
                        optimize_interval=10, topic_threshold=0.0,
                        enable_mallet=True)

    print('data')
    print(gmte.data[0])
    print('data_words')
    print(gmte.data_words[0])
    print('data_words_bigrams')
    print(gmte.data_words_bigrams[0])
    print('data_words_trigrams')
    print(gmte.data_words_trigrams[0])
    print('data_words_tetragrams')
    print(gmte.data_words_tetragrams[0])
    print('data_words_pentagrams')
    print(gmte.data_words_pentagrams[0])
    # print('data_words_nostops')
    # print(gmte.data_words_nostops[0])
    # print('data_postagged')
    # print(gmte.data_postagged[0])
    # print('data_postagged_lemmatized')
    # print(gmte.data_postagged_lemmatized[0])
    print('data_lemmatized')
    print(gmte.data_lemmatized[0])
    print('data_words_nostops')
    print(gmte.data_words_nostops[0])
    # sys.exit('die')
    # Can take a long time to run.
    gmte.compute_coherence_values(start=2, limit=55, step=5,
                                  # passes=10,
                                  iterations=500,
                                  optimize_interval=10,
                                  topic_threshold=0.0,
                                  enable_mallet=True)

    df_topic_sents_keywords = gmte.format_topics_sentences()

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                                 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)

    gmte.get_topic_distribution()
