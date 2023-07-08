import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df_raw = pd.read_csv("data/listings.csv")
df = df_raw.copy()
m, n = df.shape
print("Number of samples before cleaning: ", m)
print("Number of features before cleaning: ", n)
df.host_response_time.value_counts(normalize=True)

#delete meaningfulless features, "id" keeps for merging with reviews.csv
drop_cols0 = ["listing_url", "scrape_id", "last_scraped", "picture_url", "host_id", "host_url",
              "host_thumbnail_url", "host_picture_url", "calendar_updated", "calendar_last_scraped", "source"] # , "host_neighbourhood", "first_review", "last_review"
# availability is not relevant to the rating so delete all relevant fields. It's related to the fact that if customer could book the room
drop_cols10 = ['has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365']

# e.g. keep room type instead of property type
# e.g. keep neighbourhood_cleansed instead of neighbourhood, neighbourhood_group_cleansed which has lots of NaN value
# e.g. keep neighbourhood_cleansed instead of latitude, longtitude, all of them are geographic information
drop_cols11 = ["neighbourhood", "neighbourhood_group_cleansed", "neighborhood_overview", "property_type"] #"latitude", "longitude",
# e.g. keep minimum_nights and maximum_nights, delete minimum_minimum_nights / maximum_minimum_nights / minimum_maximum_nights / maximum_maximum_nights / minimum_nights_avg_ntm / maximum_nights_avg_ntm
drop_cols12 = ["minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm"]
print("host v.s. host total", sum((df.host_listings_count == df.host_total_listings_count) == False)) # 2918
# Most host_listings_count equals to host_total_listings_count, host_total_listings_count usually is greater than host_listings_count
print("host v.s. calculated", sum((df.host_listings_count == df.calculated_host_listings_count) == False)) # 1167
# Most host_listings_count equals to calculated_host_listings_count, host_listings_count usually is greater than calculated_host_listings_count
# calculated_host_listings_count = calculated_host_listings_count_entire_homes + calculated_host_listings_count_private_rooms + calculated_host_listings_count_shared_rooms
# These features are relevant, we could keep host_listings_count only
drop_cols13 = ["host_total_listings_count", "calculated_host_listings_count", "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms"]
# keep reviews_per_month, number_of_reviews delete number_of_reviews_ltm,number_of_reviews_l30d
drop_cols14 = ['number_of_reviews_ltm','number_of_reviews_l30d']
# delet text type features and keep 1 feature in the case that serveral features contributes the similar information
drop_cols15 = ["host_name", "name", "description", "host_location", "host_about"] #"host_since", "host_response_time"

df.drop(drop_cols0, axis=1, inplace=True)
df.drop(drop_cols10, axis=1, inplace=True)
df.drop(drop_cols11, axis=1, inplace=True)
df.drop(drop_cols12, axis=1, inplace=True)
df.drop(drop_cols13, axis=1, inplace=True)
df.drop(drop_cols14, axis=1, inplace=True)
df.drop(drop_cols15, axis=1, inplace=True)


def plot_nan(text) :
    #Show the features have NaN values
    n_nan = []
    nan_cols = []
    n_cols = len(df.columns.tolist())

    for col in df.columns.tolist():
        if df[col].isna().sum() > 0:
            nan_cols.append(col)
            n_nan.append(df[col].isna().sum())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(n_nan)
    ax.set_title(text)
    ax.set_ylabel("Number of NaN Values")
    ax.set_xticks(list(range(0, len(n_nan))))
    print(len(n_nan))
    ax.set_xticklabels(nan_cols, rotation=90)
    ax.grid("on")
    plt.tight_layout()

plot_nan("(a) NaN Values before")

#delete the features which have too many NaN values
drop_cols3 = ["host_response_time","host_response_rate","host_acceptance_rate", "host_neighbourhood","bathrooms", "license"]
df.drop(drop_cols3, axis=1, inplace=True)

#delete the samples whose review scores are NaN values
nan_index = df[df["review_scores_rating"].isna()].index
df.drop(nan_index, inplace = True)
nan_index1 = df[df["review_scores_accuracy"].isna()].index
df.drop(nan_index1, inplace = True)
nan_index2 = df[df["review_scores_cleanliness"].isna()].index
df.drop(nan_index2, inplace = True)
nan_index3 = df[df["review_scores_checkin"].isna()].index
df.drop(nan_index3, inplace = True)
nan_index4 = df[df["review_scores_communication"].isna()].index
df.drop(nan_index4, inplace = True)
nan_index5 = df[df["review_scores_location"].isna()].index
df.drop(nan_index5, inplace = True)
nan_index6 = df[df["review_scores_value"].isna()].index
df.drop(nan_index6, inplace = True)

#Assume bathroom_text empty value equals to the quantity of bedrooms, only 2 empty valus
df['bathrooms_text'].fillna(df["bedrooms"], inplace=True)
#For empty bedrooms and beds value, assume the quantity of bedrooms equals to beds
df['bedrooms'].fillna(df["beds"], inplace=True)
df['beds'].fillna(df["bedrooms"], inplace=True)
#If both bedrooms and beds are empty, assume they equals to the accommodates / 2
df['bedrooms'].fillna((df["accommodates"] / 2), inplace=True)
df['beds'].fillna(df["bedrooms"], inplace=True)

#Covert bathrooms_text to number
def clean_bathrooms_text(bathrooms_text):
    bathrooms_text_list = []
    for value in bathrooms_text:
        bathroom_text = value.split(' ')
        bathrooms_text_list.append(bathroom_text[0])

    bathroomsSer = pd.Series(bathrooms_text_list)
    return bathroomsSer

#Convert bathrooms_text whose value is "half-bath" to 0.5
df.loc[(df['bathrooms_text'] == "Shared half-bath") | (df['bathrooms_text'] == "Half-bath") | (df['bathrooms_text'] == "Private half-bath"), "bathrooms_text"] = '0.5'
df.bathrooms_text = df.bathrooms_text.apply(lambda x: float(x.split(" ")[0] if type (x) == str else str (x)))

print(df.head())
# Replacing columns with f/t with 0/1
df.replace({'f': 0, 't': 1}, inplace=True)

#remove $ in price
df.price = df.price.str[1:-3]
df.price = df.price.str.replace(",", "")
df.price = df.price.astype('int64')


# Converting to datetime
df.host_since = pd.to_datetime(df.host_since)
# Calculating the number of days since host is active
df['host_days_active'] = (pd.datetime(2022, 9, 12) - df.host_since).astype('timedelta64[D]')
# Use host_days_active instead of host_since
df.drop('host_since', axis=1, inplace=True)
df.first_review = pd.to_datetime(df.first_review)
df['first_review_since'] = (pd.datetime(2022, 9, 12) - df.first_review).astype('timedelta64[D]')
df.last_review = pd.to_datetime(df.last_review)
df['last_review_since'] = (pd.datetime(2022, 9, 12) - df.last_review).astype('timedelta64[D]')
df.drop('first_review', axis=1, inplace=True)
df.drop('last_review', axis=1, inplace=True)

# Plotting the distribution of numerical and boolean categories
df.hist(figsize=(20, 20))

# amenities
# Example of amenities listed
# print(df.amenities.loc[:1].values)
# Creating a set of all possible amenities
amenities_list = list(df.amenities)
amenities_list_string = " ".join(amenities_list)
amenities_list_string = amenities_list_string.replace('{', '')
amenities_list_string = amenities_list_string.replace('}', ',')
# amenities_list_string = amenities_list_string.replace('[', '')
amenities_list_string = amenities_list_string.replace('] [', '\',\'')
amenities_list_string = amenities_list_string.replace('"', '')
amenities_set = [x.strip() for x in amenities_list_string.split(',')]
amenities_set = set(amenities_set)
print(amenities_set)

#Create new features based on feature "amenities"
df.loc[df['amenities'].str.contains('Air conditioning|air conditioning'), 'air_conditioning'] = 1
df.loc[df['amenities'].str.contains('sound system'), 'electronics'] = 1
df.loc[df['amenities'].str.contains('BBQ grill|Fire pit'), 'bbq'] = 1
df.loc[df['amenities'].str.contains('Kitchen|kitchen'), 'kitchen'] = 1
df.loc[df['amenities'].str.contains('balcony'), 'balcony'] = 1
df.loc[df['amenities'].str.contains('Bed linens'), 'bed_linen'] = 1
df.loc[df['amenities'].str.contains('Breakfast'), 'breakfast'] = 1
df.loc[df['amenities'].str.contains('TV'), 'tv'] = 1
df.loc[df['amenities'].str.contains('Coffee maker|Nespresso machine'), 'coffee_machine'] = 1
df.loc[df['amenities'].str.contains('Cooking basics'), 'cooking_basics'] = 1
df.loc[df['amenities'].str.contains('Dishwasher|Dryer|dryer|Washer|washer'), 'white_goods'] = 1
df.loc[df['amenities'].str.contains('Elevator'), 'elevator'] = 1
df.loc[df['amenities'].str.contains('Heating|heating'), 'heating'] = 1
df.loc[df['amenities'].str.contains('Gym|gym'), 'gym'] = 1
df.loc[df['amenities'].str.contains('Children|crib|baby'), 'child_friendly'] = 1
df.loc[df['amenities'].str.contains('parking'), 'parking'] = 1
df.loc[df['amenities'].str.contains('garden|Outdoor'), 'outdoor_space'] = 1
df.loc[df['amenities'].str.contains('Host greets you'), 'host_greeting'] = 1
df.loc[df['amenities'].str.contains('Hot tub|Bathtub|hot tub|Sauna|sauna|Pool|pool'), 'tub_sauna_or_pool'] = 1
df.loc[df['amenities'].str.contains('wifi|Wifi|Ethernet connection'), 'internet'] = 1
df.loc[df['amenities'].str.contains('Long term stays allowed'), 'long_term_stays'] = 1
df.loc[df['amenities'].str.contains('Pets allowed'), 'pets_allowed'] = 1
df.loc[df['amenities'].str.contains('Private entrance'), 'private_entrance'] = 1
df.loc[df['amenities'].str.contains('Safe|Security cameras on property'), 'secure'] = 1
df.loc[df['amenities'].str.contains('Self check-in'), 'self_check_in'] = 1
df.loc[df['amenities'].str.contains('Suitable for events'), 'event_suitable'] = 1
df.loc[df['amenities'].str.contains('alarm|Fire extinguisher'), 'fire_fighting_measures'] = 1
df.loc[df['amenities'].str.contains('First aid kit'), 'first_aid_kit'] = 1
df.loc[df['amenities'].str.contains('Dedicated workspace'), 'dedicated_workspace'] = 1
df.loc[df['amenities'].str.contains('Lockbox|Lock|lock'), 'lock'] = 1
df.loc[df['amenities'].str.contains('Indoor Fireplace'), 'indoor_fireplace'] = 1
df.loc[df['amenities'].str.contains('Essentials|essentials'), 'essentials'] = 1
df.loc[df['amenities'].str.contains('Board games'), 'board_games'] = 1

# Replacing nulls with zeros for new columns
cols_to_replace_nulls = df.iloc[:,23:].columns
df[cols_to_replace_nulls] = df[cols_to_replace_nulls].fillna(0)

# Produces a list of amenity features where one category (true or false) contains fewer than 10% of listings
infrequent_amenities = []
for col in df.iloc[:,23:].columns:
    if df[col].sum() < len(df)/10:
        infrequent_amenities.append(col)
print(infrequent_amenities)

# Dropping infrequent amenity features
df.drop(infrequent_amenities, axis=1, inplace=True)

# Dropping the original amenity feature
df.drop('amenities', axis=1, inplace=True)

# host_verifications
#Create new features based on feature "amenities"
df.loc[df['host_verifications'].str.contains('phone'), 'host_verifications_phone'] = 1
df.loc[df['host_verifications'].str.contains('email'), 'host_verifications_email'] = 1
df.loc[df['host_verifications'].str.contains('work_email'), 'host_verifications_work_email'] = 1

# Replacing nulls with zeros for new columns
cols_to_replace_nulls = df.iloc[:,50:].columns
df[cols_to_replace_nulls] = df[cols_to_replace_nulls].fillna(0)

# Produces a list of host_verifications features where one category (true or false) contains fewer than 10% of listings
infrequent_host_verifications = []
for col in df.iloc[:,50:].columns:
    if df[col].sum() < len(df)/10:
        infrequent_host_verifications.append(col)
print(infrequent_host_verifications)

# Dropping infrequent host_verifications features
df.drop(infrequent_host_verifications, axis=1, inplace=True)

# Dropping the original host_verifications feature
df.drop('host_verifications', axis=1, inplace=True)

# Dropping the features that most hosts have
df.drop('host_verifications_phone', axis=1, inplace=True)
df.drop('host_has_profile_pic', axis=1, inplace=True)


# print(df.dtypes)

# # Checking the distributions of the review ratings columns
# variables_to_plot = list(df.columns[df.columns.str.startswith("review_scores") == True])
# fig = plt.figure(figsize=(12,8))
# for i, var_name in enumerate(variables_to_plot):
#     ax = fig.add_subplot(3,3,i+1)
#     df[var_name].hist(bins=10,ax=ax)
#     ax.set_title(var_name)
# fig.tight_layout()


print(df.select_dtypes('object'))
### Conducting one-hot encoding for
colist = df.select_dtypes('object')
for column in colist:
    oh = pd.get_dummies(df[column], prefix=[column])
    df = df.join(oh)
    df = df.drop(column, axis =1)

df.to_csv("data/listings_cleaned_data.csv", index=False)
df.drop('id', axis=1, inplace=True)

# drop_cols_corr = ["review_scores_accuracy", "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location", "review_scores_value"]
# corr_df = df.drop(drop_cols_corr, axis=1)
# Get a correlation matrix
# corr = corr_df.corr()
corr = df.corr()

# Look at variables correlating with our response variable
corr_y = corr['review_scores_location']

# # Plot a horizontal bar chart of the features with > 0.01 correlation (either positive or negative)
# fontsize = 10
plt.figure(figsize=(15,10))
corr_y[np.abs(corr_y) > 0.1].sort_values(ascending=False).plot.barh()
# Plot a cmap
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center='light', as_cmap=True)
plt.figure(figsize=(15,10))
# sns.heatmap(corr_df.corr(), center=0, annot=True, fmt='.2f', square=True, cmap=cmap)
sns.heatmap(df.corr(), center=0, annot=True, annot_kws={'size':5}, fmt='.2f', square=True, cmap=cmap)

# df[['number_of_reviews', 'number_of_reviews_ltm','number_of_reviews_l30d']].hist()

m, n = df.shape
print("Number of samples after cleaning: ", m)
print("Number of features after cleaning: ", n)

# plot_nan("(b) NaN Values after")

plt.show()