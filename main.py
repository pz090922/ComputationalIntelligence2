import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# import text2emotion as te
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
from sklearn.cluster import KMeans
from itertools import combinations
import networkx as nx
from sklearn.preprocessing import MinMaxScaler


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(tweet):
    sentiment = analyzer.polarity_scores(tweet)
    return sentiment['compound']


nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

file_path = './FIFA.csv'
data = pd.read_csv(file_path)
step_size = 1000000
posts_per_step = 10000
# start_index = 0
# subset_list = []
# while start_index < len(data):
#     subset = data.iloc[start_index:start_index + posts_per_step]
#     subset_list.append(subset)
#     start_index += step_size
#
# data = pd.concat(subset_list, ignore_index=True)
data = data[-1000:]
print(data)
data['Timestamp'] = pd.to_datetime(data['Date'])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
additional_stop_words = ["also", '’', '\"', '“', '”', "v", "vs"]
stop_words.update(additional_stop_words)
# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
#     return tokens



# ---- WordNet i bag_of_words, tokenize, word frequency ----
# words_no_stopwords = [word for word in all_tokens if word.lower() not in stop_words and word not in string.punctuation]
# word_freq = Counter(words_no_stopwords)
# most_common_words = word_freq.most_common(10)
# words, counts = zip(*most_common_words)

# plt.figure(figsize=(10, 6))
# plt.bar(words, counts)
# plt.xlabel('Słowa')
# plt.ylabel('Liczba wystąpień')
# plt.title('10 najczęściej występujących słów')
# plt.show()
#

# ----- VADER ------
# data['Sentiment'] = data['Tweet'].apply(lambda x: analyze_sentiment(x) if pd.notnull(x) else None)
# data['Date'] = data['Date'].dt.date
# daily_sentiment = data.groupby('Date')['Sentiment'].mean()
# plt.figure(figsize=(12, 6))
# plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o')
# plt.title('Zmiana Sentymentu w Czasie')
# plt.xlabel('Data')
# plt.ylabel('Średni Sentyment')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# --- grafy ---
# data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else [])
# all_words = [word for tokens in data['Clean_Tweet'] for word in tokens]
# word_freq = Counter(all_words)
# common_words = [word for word, freq in word_freq.most_common(40)]
# cooccurrence = Counter()
#
# for tokens in data['Clean_Tweet']:
#     for pair in combinations(set(tokens), 2):
#         if pair[0] in common_words and pair[1] in common_words:
#             cooccurrence[pair] += 1
#
# edges = pd.DataFrame(cooccurrence.items(), columns=['pair', 'count'])
# edges[['word1', 'word2']] = pd.DataFrame(edges['pair'].tolist(), index=edges.index)
# edges = edges.drop(columns=['pair'])
# G = nx.Graph()
# for index, row in edges.iterrows():
#     G.add_edge(row['word1'], row['word2'], weight=row['count'])
# pos = nx.spring_layout(G, k=0.4, iterations=100)  # k to skala repulsji między węzłami
#
# plt.figure(figsize=(16, 16))
# node_sizes = [word_freq[word] * 0.1 for word in G.nodes()]
# nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9)
# edge_weights = [d['weight']*1 for (u, v, d) in G.edges(data=True)]
# scaler = MinMaxScaler(feature_range=(1, 10))
# weights_normalized = scaler.fit_transform([[weight] for weight in edge_weights]).flatten()
# for (u, v), weight in zip(G.edges(), weights_normalized):
#     G[u][v]['weight'] = weight
# nx.draw_networkx_edges(G, pos, width=weights_normalized, alpha=0.7)
# nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
# plt.title('Graf współwystępowania najczęściej występujących słów w tweetach')
# plt.axis('off')
# plt.show()

# --- klastyeryzacja ----
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else '')
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['Clean_Tweet'])
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)

# Wyświetlenie tematów
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda, vectorizer.get_feature_names_out(), 4)
kmeans = KMeans(n_clusters=5, random_state=32)
data['Cluster'] = kmeans.fit_predict(X)
data["Labels"] = kmeans.labels_
#
# data['Date'] = data['Timestamp'].dt.date
# daily_sentiment = data.groupby(['Date', 'Cluster'])['Sentiment'].mean().unstack()
# feature_names = vectorizer.get_feature_names_out()  # Pobierz nazwy cech (słów)
# order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]  # Sortuj indeksy centrów klastrów
# for cluster_id in range(5):
#     print(f"Cluster {cluster_id}:\n")
#     sample_tweets = data[data['Cluster'] == cluster_id]['Tweet'].sample(5)  # Wybierz losowe 5 tweetów z klastra
#     for tweet in sample_tweets:
#         print(tweet)
#     print("\n")
# plt.figure(figsize=(12, 6))
# for cluster in daily_sentiment.columns:
#     plt.plot(daily_sentiment.index, daily_sentiment[cluster], marker='o', label=f'Cluster: {cluster}')
# plt.title('Zmiana sentymentu tweetów w czasie dla klastrów')
# plt.xlabel('Data')
# plt.ylabel('Średni sentyment')
# plt.xticks(rotation=45)
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#

# ----- WordCloud ------

# wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
#
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# ---- Text2Emotions ----

# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
#     return ' '.join(tokens)
#
# data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else '')
#
# # Analiza emocji
# def analyze_emotions(text):
#     emotions = te.get_emotion(text)
#     return emotions
#
# data['Emotions'] = data['Clean_Tweet'].apply(lambda x: analyze_emotions(x) if x else None)
# #
# # Przekształcenie danych emocji do DataFrame
# emotion_columns = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
# for emotion in emotion_columns:
#     data[emotion] = data['Emotions'].apply(lambda x: x[emotion] if x else 0)
#
# # Sumowanie emocji dla wszystkich tweetów
# emotion_sums = data[emotion_columns].sum()
#
# # Wizualizacja wyników
# plt.figure(figsize=(10, 6))
# sns.barplot(x=emotion_sums.index, y=emotion_sums.values/10000)
# plt.title('Sum of Emotions in Tweets')
# plt.xlabel('Emotions')
# plt.ylabel('Sum')
# plt.show()

"""
Aby wykonać analizę emocji z użyciem biblioteki text2emotion, musisz najpierw zainstalować bibliotekę text2emotion i zaimportować ją do swojego projektu. Poniżej znajduje się przykładowy kod, który pokazuje, jak zintegrować text2emotion z przetwarzaniem tekstu, analizą emocji i wizualizacją wyników.
Krok 1: Instalacja biblioteki text2emotion

Najpierw zainstaluj bibliotekę text2emotion za pomocą pip:

bash

pip install text2emotion

Krok 2: Importowanie i użycie text2emotion

Poniżej znajduje się kompletny kod, który pokazuje, jak używać text2emotion do analizy emocji w tweetach oraz jak wizualizować wyniki za pomocą wykresu słupkowego.

python

import pandas as pd
import text2emotion as te
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

# Wczytywanie danych
file_path = './FIFA.csv'
data = pd.read_csv(file_path)

# Przetwarzanie daty
data['Timestamp'] = pd.to_datetime(data['Date'])

# Przetwarzanie tekstu
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
additional_stop_words = ["also", '’', '\"', '“', '”', "v", "vs"]
stop_words.update(additional_stop_words)

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else '')

# Analiza emocji
def analyze_emotions(text):
    emotions = te.get_emotion(text)
    return emotions

data['Emotions'] = data['Clean_Tweet'].apply(lambda x: analyze_emotions(x) if x else None)

# Przekształcenie danych emocji do DataFrame
emotion_columns = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
for emotion in emotion_columns:
    data[emotion] = data['Emotions'].apply(lambda x: x[emotion] if x else 0)

# Sumowanie emocji dla wszystkich tweetów
emotion_sums = data[emotion_columns].sum()

# Wizualizacja wyników
plt.figure(figsize=(10, 6))
sns.barplot(x=emotion_sums.index, y=emotion_sums.values)
plt.title('Sum of Emotions in Tweets')
plt.xlabel('Emotions')
plt.ylabel('Sum')
plt.show()

"""