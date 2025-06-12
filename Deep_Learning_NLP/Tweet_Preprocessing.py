import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spellchecker import SpellChecker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

from dateutil.parser import parse
import multiprocessing
import pathlib
from scipy import signal
from scipy.io import wavfile
import re
import string
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

TWEETPROC_POS_DICT = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

TWEETPROC_EMOJI_PATTERN = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)

TWEETPROC_LEMMATIZER = WordNetLemmatizer()
TWEETPROC_STOP_WORDS = set(stopwords.words('english'))
TWEETPROC_SPELL_CHECKER = SpellChecker()

def Tweetproc_Get_Wordnet_Pos(words):
    """Map NLTK POS tags to WordNet POS tags"""
    word_tag_list = nltk.pos_tag(words)
    tags = []
    for tag in word_tag_list:
        if tag[1][0] in TWEETPROC_POS_DICT:
            tags.append(TWEETPROC_POS_DICT[tag[1][0]])
        else:
            tags.append(wordnet.NOUN)
    
    return tags

class TweetPreprocessor:
    # constructor: initialize with a single tweet text
    def __init__(self, tweet, index=0, special_words=[], special_chars=[],
                 replace_dict=None, spell_check_limit=3, short_word_limit=1):
        self.original_text = tweet
        self.preprocessed_text = tweet
        self.index = index
        self.tokens = []
        self.tokens_count = 0
        self.token_updated = False
        self.hashtags = ""
        self.special_words = special_words
        self.special_chars = special_chars
        self.replace_dict = replace_dict
        self.spell_check_limit = spell_check_limit
        self.spell_error_count = 0
        self.spell_error_words = []
        self.short_word_limit = short_word_limit
        self.unknown_words = []
        self.is_retweet = False

    def lowercase_text(self):
        self.preprocessed_text = self.preprocessed_text.lower()
        self.token_updated = False 
        
    
    def remove_retweet_from_text(self):
        start_text = self.preprocessed_text[:3]
        if start_text == 'rt ' or start_text == 'RT ':
            self.is_retweet = True
            self.preprocessed_text = self.preprocessed_text[3:]
            self.token_updated = False
        
    
    def remove_emojis_from_text(self):        
        self.preprocessed_text = TWEETPROC_EMOJI_PATTERN.sub(r'', self.preprocessed_text)
        self.token_updated = False
        
    
    def remove_urls_from_text(self):
        self.preprocessed_text = re.sub(r'http\S+|www\S+|https\S+', '', self.preprocessed_text, flags=re.MULTILINE)
        self.token_updated = False
    
    def remove_mentions_from_text(self):
        self.preprocessed_text = re.sub(r'@\w+', '', self.preprocessed_text)
        self.token_updated = False
    
    
    def remove_extra_spaces_from_text(self):
        self.preprocessed_text = re.sub(r'\s+', ' ', self.preprocessed_text).strip()
        self.token_updated = False
    
    def remove_punctuations_from_text(self):
        self.preprocessed_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', self.preprocessed_text)
        self.token_updated = False
    
    def extract_and_remove_hashtags_from_text(self):
        hashtags = re.findall(r'\#\w+', self.preprocessed_text)
        hashtag_words = [hashtag[1:].lower() for hashtag in hashtags]
        self.hashtags = ', '.join(hashtag_words)
        self.preprocessed_text = re.sub(r'\#\w+', '', self.preprocessed_text)
        self.token_updated = False
    
    def remove_stopwords_from_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)            

        self.tokens = [word for word in self.tokens if word not in TWEETPROC_STOP_WORDS]
        self.preprocessed_text = ' '.join(self.tokens)
        self.token_updated = True
    
    def remove_special_characters_from_text(self):
        for char in self.special_chars:
            self.preprocessed_text = self.preprocessed_text.replace(char, ' ')
        self.token_updated = False

    def remove_short_words_from_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)

        self.tokens = [word for word in self.tokens if len(word) > self.short_word_limit]
        self.preprocessed_text = ' '.join(self.tokens)
        self.token_updated = True

    def correct_spelling_in_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)
        corrected_tokens = []
        for word in self.tokens:
            if TWEETPROC_SPELL_CHECKER.unknown([word]) and (word not in self.special_words):  # specific words are skipped
                # store the work with spelling error
                self.spell_error_count += 1
                self.spell_error_words.append(word)
                # Get the most probable correction
                correction_candidates = TWEETPROC_SPELL_CHECKER.candidates(word)
                if correction_candidates and len(correction_candidates) <= self.spell_check_limit:   # too many candidates shall be ignored
                    # for simplicity just use the first candidate
                    corrected_tokens.append(correction_candidates.pop())
                else:
                    corrected_tokens.append(word)
                    self.unknown_words.append(word)  # keep track of unknown words
            else:
                corrected_tokens.append(word)
        self.token_updated = True
        self.tokens = corrected_tokens
        self.preprocessed_text = ' '.join(corrected_tokens)
        
    def remove_spelling_errors_from_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)
        corrected_tokens = []
        for word in self.tokens:
            if TWEETPROC_SPELL_CHECKER.unknown([word]) and (word not in self.special_words):  # specific words are skipped
                # store the work with spelling error
                self.spell_error_count += 1
                self.spell_error_words.append(word)
            else:
                # keep only correctly spelled words
                corrected_tokens.append(word)
        self.token_updated = True
        self.tokens = corrected_tokens
        self.preprocessed_text = ' '.join(corrected_tokens)
    
    def remove_unknown_words_from_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)

        self.tokens = [word for word in self.tokens if word not in self.unknown_words]
        self.preprocessed_text = ' '.join(self.tokens)
        self.token_updated = True
    
    def remove_numbers_from_text(self):
        self.preprocessed_text = re.sub(r'\d+', '', self.preprocessed_text)
        self.token_updated = False
    
    def replace_special_words_in_text(self):
        for word, replacement in self.replace_dict.items():
            self.preprocessed_text = re.sub(r'\b' + re.escape(word) + r'\b', replacement, self.preprocessed_text)
        self.token_updated = False
    
    def lemmatize_text(self):
        if not self.token_updated:
            self.tokens = word_tokenize(self.preprocessed_text)

        tags = Tweetproc_Get_Wordnet_Pos(self.tokens)
        # lemmatize each token based on its POS tag
        self.tokens = [TWEETPROC_LEMMATIZER.lemmatize(word, tag) for word, tag in zip(self.tokens, tags)]
        self.preprocessed_text = ' '.join(self.tokens)
        self.token_updated = True

    def process_text(self, func_list=[], verbose=0):
        for func_name in func_list:
            if hasattr(self, func_name):
                method = getattr(self, func_name)
                method()
                if verbose >= 2:
                    print(f"After {func_name}: {self.preprocessed_text}")
            else:
                raise ValueError(f"Method {func_name} does not exist in TweetPreprocessor class.")
        if not self.token_updated:
            # Tokenize the final preprocessed text
            self.tokens = word_tokenize(self.preprocessed_text)

        self.tokens_count = len(self.tokens)
        if verbose >= 2:
            print(f"processing finished for tweet {self.index}")
        return self.preprocessed_text
    
class TweetPreprocessorFactory:
    def __init__(self, tweets_dataframe, special_words=[], special_chars=[],
                 func_list=[], tweets_list = [], col_name_orig_text='original_text', 
                 n_jobs = -1):        
        if tweets_dataframe is not None:
            self.tweets_dataframe = tweets_dataframe.copy()
        elif len(tweets_list) > 0:
            self.tweets_list = tweets_list.copy()
            self.tweets_dataframe = pd.DataFrame({'original_text': tweets_list})
        else:
            raise ValueError("Either tweets_dataframe or tweets_list must be provided.")

        self.tweets_count = len(self.tweets_dataframe)
        self.cpu_count = multiprocessing.cpu_count()
        if n_jobs == -1 or n_jobs > self.cpu_count:
            self.n_jobs = self.cpu_count
        elif n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer or -1 for all available CPUs.")
        else:
            self.n_jobs = n_jobs

        # Divide tweets_count equally into n_jobs parts
        base_size = self.tweets_count // self.n_jobs
        remainder = self.tweets_count % self.n_jobs
        self.job_slices = []
        start = 0
        for i in range(self.n_jobs):
            end = start + base_size + (1 if i < remainder else 0)
            self.job_slices.append((start, end))
            start = end
        self.special_words = special_words
        self.special_chars = special_chars
        self.col_name_orig_text = col_name_orig_text
        self.func_list = func_list
        self.tweets_processed = []
    
    def batch_process_tweets(self, index_range, verbose=0):
        tweets_processed = []
        start_index = index_range[0]
        print(f"Starting processing of tweets from index {index_range[0]} to {index_range[1]} using {self.n_jobs} jobs.")
        for i in range(index_range[0], index_range[1]):
            tweet_text = self.tweets_dataframe.iloc[i][self.col_name_orig_text]
            tweet_index = self.tweets_dataframe.iloc[i].name
            preprocessor = TweetPreprocessor(tweet_text, index=tweet_index, 
                                             special_words=self.special_words,
                                             special_chars=self.special_chars)
            preprocessor.process_text(self.func_list, verbose=verbose)
            tweets_processed.append(preprocessor)
            if (verbose >= 1) and ((i - start_index + 1) % 2000 == 0):
                print(f"processing of tweets from index {index_range[0]} to {index_range[1]} finished {i - start_index + 1} laps.")
        
        print(f"FINISHED PROCESSING of tweets from index {index_range[0]} to {index_range[1]}")
        
        return tweets_processed

    def process_all_tweets(self, verbose=0):  
        paramms = [(index_range, verbose) for index_range in self.job_slices]      
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(self.batch_process_tweets, paramms)

        for result in results:
            self.tweets_processed.extend(result)
        
        # load processed tweets into a DataFrame
        for tweet in self.tweets_processed:
            self.tweets_dataframe.at[tweet.index, 'processed_text'] = tweet.preprocessed_text
            self.tweets_dataframe.at[tweet.index, 'hashtags'] = tweet.hashtags
            self.tweets_dataframe.at[tweet.index, 'is_retweet'] = tweet.is_retweet
            self.tweets_dataframe.at[tweet.index, 'tokens_count'] = tweet.tokens_count
            self.tweets_dataframe.at[tweet.index, 'spell_error_count'] = tweet.spell_error_count
            self.tweets_dataframe.at[tweet.index, 'spell_error_words'] = ', '.join(tweet.spell_error_words)
            self.tweets_dataframe.at[tweet.index, 'unknown_words'] = ', '.join(tweet.unknown_words)
        return self.tweets_dataframe
    
    def save_processed_tweets(self, output_filename):
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
        self.tweets_dataframe.to_csv(output_filename, index=False)
        print(f"Processed tweets saved to {output_filename}")