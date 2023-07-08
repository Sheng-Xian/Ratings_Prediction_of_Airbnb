import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
import re
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, median_absolute_error


df_raw = pd.read_csv("data/listings.csv")
df = df_raw.copy()
def plot_violin() :
    fig, axes = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df, x="neighbourhood_cleansed", y="review_scores_location")
    # sns.boxplot(x=df['neighbourhood_cleansed'], y=df['review_scores_rating'], data=df)

    axes.set_title('Review scores neighbourhood by neighbourhood')

    axes.yaxis.grid(True)
    axes.set_xlabel('Neighbourhood')
    axes.set_ylabel('Review scores neighbourhood')
    plt.show()


# plot_violin()

df_cleaned = pd.read_csv("data/listings_cleaned_data.csv")
# df_cleaned = pd.read_csv("data/listings_cleaned_data_for_neighborhood.csv")


# 'latitude','longitude'
numerical_columns = ['host_listings_count', 'bathrooms_text', 'bedrooms', 'beds','accommodates', 'price',
                     'number_of_reviews','reviews_per_month', 'minimum_nights', 'maximum_nights','first_review_since',
                     'last_review_since','host_days_active', 'latitude','longitude']
scaler = StandardScaler()
# temp1 = df_cleaned.drop(columns=numerical_columns)
df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
# X = subset_neighborhood.drop(columns="review_scores_location")
# y = subset_neighborhood["review_scores_location"]
df_cleaned.drop('id', axis=1, inplace=True)


subset_neighborhood = df_cleaned[["['neighbourhood_cleansed']_Dn Laoghaire-Rathdown",
                                  "['neighbourhood_cleansed']_Dublin City", "['neighbourhood_cleansed']_Fingal",
                                  "['neighbourhood_cleansed']_South Dublin", "review_scores_location", 'latitude','longitude']] #
X = subset_neighborhood.drop(columns="review_scores_location").to_numpy()
y = subset_neighborhood["review_scores_location"].to_numpy()

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


def plot_knn_k_selection():
    k_range = [1]
    # k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mean_absolute_error = []
    mean_squared_error = []
    for k in k_range:
        classifier = KNeighborsRegressor(n_neighbors=k, weights='uniform')
        classifier.fit(Xtrain, ytrain)
        ypred = classifier.predict(Xtest)
        print("----knn----")
        # mse_score = mean_squared_error(ytest, ypred)
        r2_score = metrics.r2_score(ytest, ypred)
        print(r2_score)
        print(classifier.score(Xtrain, ytrain))
        print(classifier.score(Xtest, ytest))
    # plt.errorbar(k_range, mean_absolute_error, yerr=mean_squared_error, linewidth=3)
    # plt.xlabel('k'); plt.ylabel('F1 Score')
    # plt.title("KNeighborsClassifier: Different F1 score for different k")
    # plt.show()


def plot_linear_regression() :
    lr = LinearRegression()
    lr.fit(Xtrain, ytrain)
    ypred = lr.predict(Xtest)
    mse_score = mean_squared_error(ytest, ypred)
    r2_score = metrics.r2_score(ytest, ypred)
    print("----linear regression----")
    print(mse_score)
    print(r2_score)
    print(lr.score(Xtrain, ytrain))
    print(lr.score(Xtest, ytest))


def plot_decision_tree() :
    classifier = DecisionTreeRegressor(max_depth=2)
    classifier.fit(Xtrain, ytrain)
    ypred = classifier.predict(Xtest)
    print("----decision tree----")
    mse_score = mean_squared_error(ytest, ypred)
    r2_score = metrics.r2_score(ytest, ypred)
    print(mse_score)
    print(r2_score)
    print(classifier.score(Xtrain, ytrain))
    print(classifier.score(Xtest, ytest))


def plot_lasso() :
    # k = 5 only
    model = Lasso(alpha=1 / 2)  # <=> C = 1
    kf = KFold(n_splits=5)
    temp = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])
        temp.append(mean_squared_error(y[test], ypred))

    print("5-fold cross validation results:")
    print("Mean error = %f; Standard deviation = %f" % (np.array(temp).mean(), np.array(temp).std()))


plot_knn_k_selection()
plot_linear_regression()
plot_decision_tree()
plot_lasso()
