import re
import pandas as pd
from pprint import pprint
from Gensim_Mallet_Topic_Extractor import Gensim_Mallet_Topic_Extractor

if __name__ == '__main__':
    # freeze_support()

    gmte = Gensim_Mallet_Topic_Extractor('english')

    # Import Dataset
    df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    print(df.target_names.unique())
    df.head()

    # Convert to list
    data = df.content.values.tolist()

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    pprint(data[:1])

    gmte.extract_topics(data, 20)

    # Can take a long time to run.
    gmte.compute_coherence_values(start=2, limit=40, step=1)


    df_topic_sents_keywords = gmte.format_topics_sentences()

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)

    gmte.get_topic_distribution()
