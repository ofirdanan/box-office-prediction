import argparse
import numpy as np
import pandas as pd
import datetime
import operator
import json, requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.impute import KNNImputer
import pickle
from xgboost.sklearn import XGBRegressor



# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

#prediction code

def get_genres_list(row):
    list_genres = []
    for dict_genre in row:
        list_genres.append(dict_genre['name'])
    return list_genres

def get_day_of_year(date):
    res = '{:%m-%d}'.format(date, '%Y-%m-%d')
    return res

def get_languages_list(row):
    list_lan = []
    for language in row:
        list_lan.append(list(language.values())[1])
    return list_lan

def get_actors_list(row):
    list_actors = []
    for actor in row:
        list_actors.append(list(actor.values())[5])
    return list_actors

def get_gender_ratio(row):
    male = 0
    female = 0
    for actor in row:
        if (actor['gender'] == 2):
            male += 1
        elif (actor['gender'] == 1):
            female += 1
    if (female == 0):
        return 1
    else:
        return male/female
    
def get_depar_num_workers(row):
    depar_num_workers = {}
    for depar in row:
        depar_name = depar['department']
        if depar_name not in depar_num_workers.keys():
            depar_num_workers[depar_name] = 1
        else:
            depar_num_workers[depar_name] += 1
    return depar_num_workers

def get_crew_list(row, topic):
    res_list = []
    for crew_member in row:
        if crew_member['job'] == topic:
            res_list.append(crew_member['name'])
    return res_list

def get_production_companies_list(row):
    list_companies = []
    for company in row:
        list_companies.append(company['name'])
    return list_companies

def get_len_overview(row):
    if (str(row) == 'nan'):
        return 0
    else:
        text_tokens = word_tokenize(row)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        return len(tokens_without_sw)
    
def get_keywords_list(row):
    keywords_list = []
    for word in row:
        keywords_list.append(word['name'])
    return keywords_list

def get_reviews_sentiment_score(imdb_id):
    sentiment_score = 0
    url = 'https://www.imdb.com/title/' + str(imdb_id) +'/reviews?ref_=tt_ql_3'
    response = requests.get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser') 
    reviews_body = html_soup.find_all('div', class_ = 'text show-more__control')
    reviews_title = html_soup.find_all('a', class_ = 'title')
    analyser = SentimentIntensityAnalyzer()
    if len(reviews_body) == 0:
        return 0
    else:
        for review_b in reviews_body:
            sentiment_score += analyser.polarity_scores(review_b.text)['compound']
        for review_t in reviews_title:
            sentiment_score += analyser.polarity_scores(review_t.text)['compound']
        return sentiment_score/(len(reviews_title)+len(reviews_body))
    
def get_top_columns_df(data, data_columns, top_columns):
    mlb = MultiLabelBinarizer()
    df_dummy = pd.DataFrame(mlb.fit_transform(data[data_columns]), columns = mlb.classes_, index=data[data_columns].index)
    for sub in columns_dict[top_columns]:
        if(sub in df_dummy.columns):
            data[sub] =  df_dummy[sub]
        else:
            data[sub] = data.apply(lambda row: 0, axis = 1)
    return data

#Casting
data['genres'] = data['genres'].apply(lambda x: eval(x))
data['production_companies'] = data['production_companies'].apply(lambda x: eval(x))
data['production_countries'] = data['production_countries'].apply(lambda x: eval(x))
data['spoken_languages'] = data['spoken_languages'].apply(lambda x: eval(x))
data['Keywords'] = data['Keywords'].apply(lambda x: eval(x))
data['cast'] = data['cast'].apply(lambda x: eval(x))
data['crew'] = data['crew'].apply(lambda x: eval(x))
data['release_date'] = data['release_date'].apply(lambda x: 
                                                  datetime.datetime.strptime(x, '%Y-%m-%d'))

#Feature Engineering
columns_dict ={}
with open('columns_dict.pkl', 'rb') as f:
    columns_dict = pickle.load(f)

data_id = data['id'].copy()    
    
data = data.drop(columns = ['homepage', 'belongs_to_collection', 'status', 'original_title',
                           'backdrop_path', 'id', 'poster_path', 'title', 'original_language', 'tagline', 'video'])

data['log_budget'] = data.apply(lambda row: np.log(row['budget']) if row['budget'] != 0 else 0, axis = 1)
data['log_revenue'] = data.apply(lambda row: np.log(row['revenue']) if row['revenue'] != 0 else 0, axis = 1)
data = data.drop(columns = ['budget'])

print('start genres')
#Create genres binary features
data['genres'] = data['genres'].apply(lambda x: get_genres_list(x))
mlb = MultiLabelBinarizer()
df_genres = pd.DataFrame(mlb.fit_transform(data['genres']), columns = mlb.classes_, index=data['genres'].index)
data = pd.concat([data, df_genres], axis=1)
data = data.drop(columns = ['genres'])

print('start date')
#Create year and month binary features
data['year'] = pd.DatetimeIndex(data['release_date']).year
data['month'] = pd.DatetimeIndex(data['release_date']).month
data = pd.concat([data.drop(columns = ['month']), pd.get_dummies(data['month'])], axis=1)

#Create 10 days features that are most relevant to predict revenue
data['Day of year'] = data.apply(lambda row: get_day_of_year(row['release_date'].date()), axis = 1)
dummy = pd.get_dummies(data['Day of year'])
for day in columns_dict['top_10_days']:
    if(day in dummy.columns):
        data[day] =  dummy[day]
    else:
        data[day] = 0

data = data.drop(columns = ['Day of year', 'release_date'])

print('start languages')
#Create language binary features
data['languages'] = data.apply(lambda row: get_languages_list(row['spoken_languages']), axis=1)
data = get_top_columns_df(data, 'languages', 'top_10_lang')
data = data.drop(columns = ['languages', 'spoken_languages'])

print('start actors')
#Create actors features
data['actors_list'] = data.apply(lambda row: get_actors_list(row['cast']), axis=1)
data = get_top_columns_df(data, 'actors_list', 'top_15_actors')
data['actors_num'] = data.apply(lambda row: len(row['actors_list']), axis=1)
data['actors_gender_ratio'] = data.apply(lambda row: get_gender_ratio(row['cast']), axis =1)
data = data.drop(columns = ['actors_list', 'cast'])

print('start department')
#Create department size features
data['dep_production'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Production'] if 'Production' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Sound'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Sound'] if 'Sound' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Art'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Art'] if 'Art' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Writing'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Writing'] if 'Writing' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Directing'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Directing'] if 'Directing' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Editing'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Editing'] if 'Editing' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Costume_&_Make-Up'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Costume & Make-Up'] if 'Costume & Make-Up' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Camera'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Camera'] if 'Camera' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Crew'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Crew'] if 'Crew' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Visual_Effects'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Visual Effects'] if 'Visual Effects' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)
data['dep_Lighting'] = data.apply(lambda row: get_depar_num_workers(row['crew'])['Lighting'] if 'Lighting' in get_depar_num_workers(row['crew']).keys() else 0 , axis=1)

print('start crew')
#Create proucers, directors ans writers features
data['producers_list'] = data.apply(lambda row: get_crew_list(row['crew'], 'Producer'), axis=1)
data = get_top_columns_df(data, 'producers_list', 'top_producers')
data['directors_list'] = data.apply(lambda row: get_crew_list(row['crew'], 'Director'), axis=1)
data = get_top_columns_df(data, 'directors_list', 'top_directors')
data['writes_list'] = data.apply(lambda row: get_crew_list(row['crew'], 'Writer'), axis=1)
data = get_top_columns_df(data, 'writes_list', 'top_writers')
data = data.drop(columns = ['producers_list', 'directors_list', 'writes_list', 'crew'])

print('start production')
#Create production companies features
data['production_companies_list'] = data.apply(lambda row: get_production_companies_list(row['production_companies']), axis=1)
data = get_top_columns_df(data, 'production_companies_list', 'top_produ_comp')
data['production_companies_num'] = data.apply(lambda row: len(row['production_companies_list']), axis=1)
data = data.drop(columns = ['production_companies_list', 'production_companies'])

#Create production countries features
data['production_countries_list'] = data.apply(lambda row: get_production_companies_list(row['production_countries']), axis=1)
data = get_top_columns_df(data, 'production_countries_list', 'top_produ_coun')
data = data.drop(columns = ['production_countries_list', 'production_countries'])

print('start keywords')
#Create keywords features
data['keywords_list'] = data.apply(lambda row: get_keywords_list(row['Keywords']), axis = 1)
data['num_keywords'] = data.apply(lambda row: len(row['keywords_list']), axis = 1)
data = get_top_columns_df(data, 'keywords_list', 'top_keywords')
data = data.drop(columns = ['keywords_list', 'Keywords'])

#Create reviews sentiment features
print('start reviews')
data['reviews_sentiment_score'] = data.apply(lambda row: get_reviews_sentiment_score(row['imdb_id']), axis = 1)
data = data.drop(columns = ['imdb_id'])

print('start overview')
#Create overview feature
data['overview_len'] = data.apply(lambda row: get_len_overview(row['overview']), axis = 1)
data = data.drop(columns = ['overview'])

#Imputation
data['vote_count'] = data['vote_count'].replace(0, np.nan)
data['vote_average'] = data['vote_average'].replace(0, np.nan)
data['log_budget'] = data['log_budget'].replace(0, np.nan)
data['runtime'] = data['runtime'].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
data_impu = pd.DataFrame(imputer.fit_transform(data), columns = data.columns, index = data.index)

#Model
clf = XGBRegressor()
clf = pickle.load(open('model.pkl', 'rb'))
x_test = data_impu.drop(columns = ['revenue', 'log_revenue'])
y_test = data_impu['log_revenue']
y_pred_test = clf.predict(x_test)

prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data_id
prediction_df['revenue'] = np.expm1(y_pred_test)


# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


### Utility function to calculate RMSLE
def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


### Example - Calculating RMSLE
res = rmsle(data['revenue'], prediction_df['revenue'])
print("RMSLE is: {:.6f}".format(res))


