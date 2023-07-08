import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

df_raw = pd.read_csv("data/listings.csv")
df = df_raw.copy()
def plot_violin() :
    fig, axes = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df, x="host_is_superhost", y="review_scores_rating")
    # sns.boxplot(x=df['neighbourhood_cleansed'], y=df['review_scores_rating'], data=df)

    axes.set_title('Review scores rating by superhost')

    axes.yaxis.grid(True)
    axes.set_xlabel('Superhost')
    axes.set_ylabel('Review scores rating')
    plt.show()


plot_violin()