import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

folder_path = '.'  # Your folder path

dataframes = {
    'info_base_games': None,
    'gamalytic_steam_games': None,
    'demos': None,
    'dlcs': None
}

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        for key in dataframes.keys():
            if key in filename:
                file_path = os.path.join(folder_path, filename)
                dataframes[key] = pd.read_csv(file_path, low_memory=False)
                break

demos = pd.read_csv('demos.csv')
dlcs = pd.read_csv('dlcs.csv')
gamalytic_steam_games = pd.read_csv('gamalytic_cls_sales.csv')
info_base_steam_games = pd.read_csv('info_base_games.csv')
# Convert all ID columns to nullable integer type with coercion
print(gamalytic_steam_games.head())
for df, col in [
    (gamalytic_steam_games, 'steamId'),
    (info_base_steam_games, 'appid'),
    (dlcs, 'base_appid'),
    (demos, 'full_game_appid')
]:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

for df, col in [
    (gamalytic_steam_games, 'steamId'),
    (info_base_steam_games, 'appid'),
    (dlcs, 'base_appid'),
    (demos, 'full_game_appid')
]:
    df.dropna(subset=col, inplace=True)

merged_data = (
    gamalytic_steam_games
    .merge(info_base_steam_games, how='inner', left_on='steamId', right_on='appid')
    .merge(dlcs, how='left', left_on='steamId', right_on='base_appid')
    .merge(demos, how='left', left_on='steamId', right_on='full_game_appid')
)
# default_date = pd.Timestamp('2025-03-01')
# merged_data['release_date'] = merged_data['release_date'].fillna(default_date)
# merged_data.info()


# fill null
genres = {'AAA': 'Action',
          'Indie': 'Adventure, Casual, Indie',
          'Hobbyist': 'Action, Casual, Indie',
          'AA': 'Action, Indie'}

review_score = {
    "Indie": 75.24594903516439,
    "Hobbyist": 68.95856923519815,
    "AA": 78.34399308556613,
    "AAA": 78.60340632603406
}

x = {'17. Dez. 2021': 'Dec 17, 2021', '13 lutego 2023': 'Feb 13, 2023', '2016년 10월 7일': 'Oct 7,2016',
     '2024년 2월 26일': 'Feb 26, 2024', '2022 年 12 月 13 日	': 'Dec 13, 2022', 'Apr-25': 'Apr 1, 2025',
     'Feb-25': 'Feb 1, 2025', 'Mar-25': 'Mar 1, 2025', 'Jul-25': 'Jul 1, 2025', 'Q2 2025': 'May 1, 2025',
     'Q1 2025': 'Feb 1, 2025', 'Q4 2025': 'Nov 1, 2025'}
merged_data['release_date'] = merged_data['release_date'].map(lambda m: x.get(m, m))

merged_data['metacritic'] = merged_data['metacritic'].fillna(0)
merged_data['price'] = merged_data['price'].fillna(0)
merged_data['achievements_total'] = merged_data['achievements_total'].fillna(0)
merged_data['supported_platforms'] = merged_data['supported_platforms'].fillna("['windows']")
merged_data['release_date'] = merged_data['release_date'].fillna(pd.to_datetime('Mar 1, 2025', dayfirst=True))
merged_data['steam_trading_cards'] = merged_data['steam_trading_cards'].fillna(False)
merged_data['workshop_support'] = merged_data['workshop_support'].fillna(False)
merged_data['publisherClass'] = merged_data['publisherClass'].fillna('Indie')

for i in merged_data['publisherClass'].unique():
    merged_data.loc[merged_data['publisherClass'] == i, 'genres'] = merged_data.loc[
        merged_data['publisherClass'] == i, 'genres'].fillna(genres[i])
    merged_data.loc[(merged_data['publisherClass'] == i) & (merged_data['genres'] == ''), 'genres'] = genres[i]

for i in merged_data['publisherClass'].unique():
    merged_data.loc[merged_data['publisherClass'] == i, 'reviewScore'] = merged_data.loc[
        merged_data['publisherClass'] == i, 'reviewScore'].fillna(review_score[i])
    merged_data.loc[(merged_data['publisherClass'] == i) & (merged_data['reviewScore'] == ''), 'reviewScore'] = int(review_score[i])

# correct data type
merged_data['release_date'] = pd.to_datetime(merged_data['release_date'], format='mixed', errors='coerce')
merged_data['metacritic'] = merged_data['metacritic'].astype(float)
merged_data['achievements_total'] = merged_data['achievements_total'].astype(float).astype(int)
merged_data['reviewScore'] = pd.to_numeric(merged_data['reviewScore'], errors='coerce')
merged_data['reviewScore_log'] = np.log1p(merged_data['reviewScore'])

# dealing with inconsistency
merged_data['steam_achievements'] = [i > 0 for i in merged_data['achievements_total']]

# Feature Engineering


# log data
merged_data['metacritic_log'] = np.log1p(merged_data['metacritic'])
merged_data['achievements_total_log'] = np.log1p(merged_data['achievements_total'])
merged_data['reviewScore_log'] = np.log1p(merged_data['reviewScore'])
merged_data['price_log'] = np.log1p(merged_data['price'])

# feature eng
top_genres = ['Action', 'Strategy', 'Adventure', 'Casual', 'Indie', 'RPG', 'Simulation']
merged_data['sorted_genres'] = [', '.join(sorted([i.strip()
                                                  for i in genres.split(',')
                                                  if i.strip() in top_genres]))
                                if pd.notna(genres)
                                else pd.NA
                                for genres in merged_data['genres']]

for i in merged_data['publisherClass'].unique():
    merged_data.loc[(merged_data['publisherClass'] == i) & (merged_data['sorted_genres'] == ''), 'sorted_genres'] = \
    genres[i]

merged_data['has_metacritic'] = merged_data['metacritic'] != 0
merged_data['has_dlc'] = merged_data['dlc_appid'].notna()
merged_data['has_demo'] = merged_data['demo_appid'].notna()
merged_data['is_free'] = merged_data['price'] == 0
merged_data['premium_score'] = (
        merged_data['steam_trading_cards'].astype(int) +
        merged_data['workshop_support'].astype(int) +
        merged_data['has_dlc'].astype(int) +
        merged_data['has_metacritic'])
merged_data['premium_score_review_interaction'] = merged_data['premium_score'] * merged_data['reviewScore']
merged_data['metacritic_review_interaction'] = merged_data['metacritic'] * merged_data['reviewScore']
merged_data['year'] = merged_data['release_date'].dt.year
merged_data['months_passed'] = (
    (pd.to_datetime('apr 1, 2025').year - merged_data['release_date'].dt.year) * 12 +
    (pd.to_datetime('apr 1, 2025').month - merged_data['release_date'].dt.month)
)

# label encoding
le_publisher = joblib.load('publisherClass_encoder.pkl')
merged_data['encoded_publisherClass'] = le_publisher.transform(merged_data['publisherClass'].astype(str))
le_platforms = joblib.load('supported_platforms_encoder.pkl')
merged_data['encoded_supported_platforms'] = le_platforms.transform(merged_data['supported_platforms'].astype(str))
le_genres = joblib.load('sorted_genres_encoder.pkl')
merged_data['encoded_sorted_genres'] = le_genres.transform(merged_data['sorted_genres'].astype(str))
copiesSold_class = joblib.load('copiesSold_class_encoder.pkl')
merged_data['encoded_copiesSold_class'] = copiesSold_class.transform(merged_data['copiesSold'].astype(str))
merged_data["year"]=merged_data["year"].fillna(2025)
# reviewScore
merged_data = merged_data.drop(['steamId', 'aiContent', 'appid', 'name_x', 'base_appid',
                                'dlc_appid', 'name_y', 'full_game_appid',
                                'demo_appid', 'name'], axis=1)
merged_data.info()

# model loading

model = joblib.load('classification_Stacking Ensemble (LR+GB+NB)_model.pkl')
# catboost = joblib.load('regression_catboost_model.pkl')

from catboost import  CatBoostClassifier
x =[
'price_log', 'reviewScore_log', 'metacritic_log',  'steam_trading_cards', 'encoded_sorted_genres', 'encoded_publisherClass', 'premium_score_review_interaction',
      'workshop_support', 'achievements_total_log', 'has_dlc', 'has_demo', 'is_free', 'encoded_supported_platforms', 'year', 'metacritic_review_interaction'
]
X = merged_data[x]
null_counts = X.isnull().sum()
print(null_counts[null_counts > 0])

Y = merged_data['encoded_copiesSold_class']
y_pred = model.predict(X)
y_proba_test = model.predict_proba(X)
print("\n    Test Accuracy:", accuracy_score(Y, y_pred))
print("    Test Precision:", precision_score(Y, y_pred, average='weighted'))
print("    Test Recall:", recall_score(Y, y_pred, average='weighted'))
print("    Test F1:", f1_score(Y, y_pred, average='weighted'))
print("    Test AUC:", roc_auc_score(Y, y_proba_test, multi_class='ovr'))





