
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Deep_Learning_NLP")))
from Tweet_Preprocessing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    datapath = "C:/projects/tweets_covid"
    tweet_data_1 = pd.read_csv(os.path.join(datapath, "Covid-19 Twitter Dataset (Apr-Jun 2020).csv"))
    tweet_data_1_backup = tweet_data_1.copy()
    tweet_data_2 = pd.read_csv(os.path.join(datapath, "Covid-19 Twitter Dataset (Apr-Jun 2021).csv"))
    tweet_data_2_backup = tweet_data_2.copy()
    tweet_data_3 = pd.read_csv(os.path.join(datapath, "Covid-19 Twitter Dataset (Aug-Sep 2020).csv"))
    tweet_data_3_backup = tweet_data_3.copy()

    tweet_data_1.dropna(subset=['lang', 'original_text'], how='any', axis=0, inplace=True)
    tweet_data_1.drop(columns=['id', 'source', 'lang', 'created_at'], inplace=True)
    tweet_data_2.dropna(subset=['lang', 'original_text'], how='any', axis=0, inplace=True)
    tweet_data_2.drop(columns=['id', 'source', 'lang', 'created_at'], inplace=True)
    tweet_data_3.dropna(subset=['lang', 'original_text'], how='any', axis=0, inplace=True)
    tweet_data_3.drop(columns=['id', 'source', 'lang', 'created_at'], inplace=True)

    tweet_data_1['text_len'] = tweet_data_1['original_text'].apply(lambda x: len(x))
    tweet_data_2['text_len'] = tweet_data_2['original_text'].apply(lambda x: len(x))
    tweet_data_3['text_len'] = tweet_data_3['original_text'].apply(lambda x: len(x))

    #tweet_data_test = tweet_data_1.iloc[:1000, :].copy()
    special_words = ['covid', 'covid19', 'covid-19', 'coronavirus', 'corona']
    special_chars = ['\u2026']
    function_list = ['remove_retweet_from_text',
                    'remove_mentions_from_text', 
                    'lowercase_text',
                    'extract_and_remove_hashtags_from_text',
                    'remove_urls_from_text',
                    'remove_special_characters_from_text',
                    'remove_spelling_errors_from_text',  
                    'lemmatize_text',
                    'remove_punctuations_from_text',
                    'remove_unknown_words_from_text',
                    'remove_numbers_from_text',
                    'remove_stopwords_from_text'           
                    ]

    tweets_processor_1 = TweetPreprocessorFactory(tweet_data_1,
                                                special_words=special_words,
                                                special_chars=special_chars,
                                                func_list=function_list,n_jobs=-1)
    tweets_processed = tweets_processor_1.process_all_tweets()
    tweets_processor_1.save_processed_tweets("processed_tweets_1.csv")
    print(f"Everything done")