import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
import re
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

df_raw = pd.read_csv("data/listings.csv")
df = df_raw.copy()
def plot_violin() :
    fig, axes = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df, x="neighbourhood_cleansed", y="review_scores_location")
    axes.set_title('Review scores neighbourhood by neighbourhood')
    axes.yaxis.grid(True)
    axes.set_xlabel('Neighbourhood')
    axes.set_ylabel('Review scores neighbourhood')
    plt.show()

#plot_violin()

df_cleaned = pd.read_csv("data/listings_cleaned_data.csv")
# df_cleaned = pd.read_csv("data/listings_cleaned_data_for_neighborhood.csv")

numerical_columns = ['host_listings_count', 'bathrooms_text', 'bedrooms', 'beds','accommodates', 'price',
                     'number_of_reviews','reviews_per_month', 'minimum_nights', 'maximum_nights','first_review_since',
                     'last_review_since','host_days_active', 'latitude','longitude']
scaler = StandardScaler()
# temp1 = df_cleaned.drop(columns=numerical_columns)
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
# X = subset_neighborhood.drop(columns="review_scores_location")
# y = subset_neighborhood["review_scores_location"]

def formatting(value):
    if value < 4.6:
        return -1
    elif value < 4.82:
        return 0
    elif value < 5:
    # if value < 5:
        return 1
    else:
        return 2


df_cleaned['review_scores_location'] = df_cleaned['review_scores_location'].apply(formatting)
df_cleaned.drop('id', axis=1, inplace=True)

subset_neighborhood = df_cleaned[["['neighbourhood_cleansed']_Dn Laoghaire-Rathdown",
                                  "['neighbourhood_cleansed']_Dublin City", "['neighbourhood_cleansed']_Fingal",
                                  "['neighbourhood_cleansed']_South Dublin", "review_scores_location", 'latitude','longitude']] #

df_cleaned.to_csv("test.csv")

X = subset_neighborhood.drop(columns="review_scores_location")
y = subset_neighborhood["review_scores_location"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


def plot_knn_k_selection():
    k_range = [1]
    # k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 30]
    mean_error = []
    std_error = []
    for k in k_range:
        classifier = KNeighborsClassifier(n_neighbors=k,metric="minkowski") #weights='uniform'
        classifier.fit(Xtrain, ytrain)
        ypred = classifier.predict(Xtest)
        print("-----knn-----")
        print("Test Set Classification report: ", classification_report(ytest, ypred))
        print("Test Set Accuracy: ", accuracy_score(ytest, ypred))
        # from sklearn.model_selection import cross_val_score
    #     scores = cross_val_score(classifier, X, y, cv=5, scoring='f1')
    #     mean_error.append(np.array(scores).mean())
    #     std_error.append(np.array(scores).std())
    # plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    # plt.xlabel('k'); plt.ylabel('F1 Score')
    # plt.title("KNeighborsClassifier: Different F1 score for different k")
    # plt.show()


def plot_logistic_regression() :
    # lr = LogisticRegression(penalty='none',solver='newton-cg', max_iter=1000)
    lr = OneVsOneClassifier(LogisticRegression(C=1.0, tol=1e-6, max_iter=1000))
    lr.fit(Xtrain, ytrain)
    print("----logistic regression----")
    print("Training Set score: ", lr.score(Xtrain, ytrain))
    print("Test Set score: ", lr.score(Xtest, ytest))

def plot_dummy() :
    classifier = DummyClassifier(strategy='most_frequent').fit(Xtrain, ytrain)
    classifier.fit(Xtrain, ytrain)
    print("----Baseline Classifier: Most frequent----")
    print("Training Set score: ", classifier.score(Xtrain, ytrain))
    print("Test Set score: ", classifier.score(Xtest, ytest))


plot_logistic_regression()
plot_knn_k_selection()
plot_dummy()

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


def tokenize(text) :

    tokens = nltk.word_tokenize(text)
    # from nltk.corpus import stopwords
    # without_stopwords = [w for w in tokens if not w in stopwords.words('english')]
    stems = []
    for token in tokens:
        stems.append((SnowballStemmer('english').stem(token))) # for token in without_stopwords
    from nltk.stem import PorterStemmer
    # stemmer = PorterStemmer()
    # stems = [stemmer.stem(token) for token in tokens]
    return stems


# df_cleaned['neighborhood_overview'] = df_cleaned['neighborhood_overview'].astype(str)
# df_cleaned['neighborhood_overview'] = df_cleaned['neighborhood_overview'].apply(lambda x: demoji(x))
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(tokenizer=tokenize, max_df=0.01, min_df=1,ngram_range=(2, 2), max_features=50)
# text = df_cleaned["neighborhood_overview"].astype(str).values
# X = vectorizer.fit_transform(text)
# X = X.toarray()
# names = vectorizer.get_feature_names()
# comments_dataframe = pd.DataFrame(X)
# comments_dataframe.columns=names
# comments_dataframe.to_csv("data/comments_vector_neighborhood.csv", index=False)

comments_vector = pd.read_csv("data/comments_vector_neighborhood.csv")
# comments_columns = ['bakeri', 'bay', 'bray', 'breakfast', 'capel', 'convent', 'glasnevin', 'ground', 'hall', 'harbour', 'ifsc', 'merrion', 'parnel', 'univers']
# temp_df = comments_vector[comments_columns]
subset_neighborhood = subset_neighborhood.join(comments_vector)

X = subset_neighborhood.drop(columns="review_scores_location")
y = subset_neighborhood["review_scores_location"]


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


plot_logistic_regression()
plot_knn_k_selection()
plot_dummy()
