import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
import re
from googletrans import Translator
import time

df_raw = pd.read_csv('data/reviews_with_language.csv', encoding='utf-8')
df = df_raw.copy()

def demoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U00010000-\U0010ffff"
                               "]+", flags=re.UNICODE)
    temp = emoji_pattern.sub(r'', text)
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~€]+'
    return (re.sub(remove_chars, '', temp))


df['comments'] = df['comments'].astype(str)
df['comments'] = df['comments'].apply(lambda x: demoji(x))


index = df[df['language'] != 'en'].index
df.drop(index, inplace = True)
df.comments = df.comments.str.replace("<br/>", "")
# df.to_csv("data/reviews_english_only.csv", index=False)

#only keep listing_id and comments
df = df.loc[:, ['listing_id', 'comments']]
#count the quantity of reviews
df['count'] = np.ones((df.shape[0],1), dtype=np.int8)
df['comments'] = df['comments'].apply(str)
reviews_by_listing_id = df.groupby(by = "listing_id")['comments'].apply(lambda x: " ".join(x))
reviews_by_listing_id = pd.DataFrame(reviews_by_listing_id)
reviews_by_listing_id['count'] = df.groupby(by = "listing_id")['count'].sum()
reviews_by_listing_id = reviews_by_listing_id.reset_index()


reviews_by_listing_id.to_csv("data/reviews_by_listing_id.csv", index=False)


def tokenize(text) :
    tokens = nltk.word_tokenize(text)
    # from nltk.corpus import stopwords
    # without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    # stems = []
    # for token in tokens:
    #     stems.append((SnowballStemmer('english').stem(token))) # for token in without_stopwords
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]
    return stems

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.1, min_df=1,ngram_range=(1, 1), max_features=5000)
comments_text = reviews_by_listing_id["comments"].astype(str).values
X = vectorizer.fit_transform(comments_text)
print("\t> Names:", vectorizer.get_feature_names())
print("\t> Number:", len(vectorizer.get_feature_names()))
print("\n> Total number of comments:", len(comments_text))
X = X.toarray()
names = vectorizer.get_feature_names()
comments_dataframe = pd.DataFrame(X)
comments_dataframe.columns=names
comments_dataframe.to_csv("comments_vector.csv", index=False)

# Merge the reviews and listings in the listings.csv.
df = pd.read_csv("data/listings.csv")
df1 =df[['listings_id', 'reviews_scores_rating', 'scores_location']]
df2 = reviews_by_listing_id[['listings_id', 'review_id', 'comments']]
sample1_rh_df = pd.merge(df1, df2, on="listings_id")

set(df1.home_id) - set(sample1_rh_df.home_id)

sample2_rh_df = sample1_rh_df
print('*' * 40 + '\nThere are:\n' + '-' * 40)
print(str(len(sample2_rh_df.groupby('home_id'))) + " Airbnb homes in total.\n" + '-' * 40)
print(str(len(sample2_rh_df)) + " reviews in total.\n" + '-' * 40)