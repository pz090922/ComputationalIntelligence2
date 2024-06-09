import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# import text2emotion as te
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
step_size = 10000
posts_per_step = 1000
start_index = 0
subset_list = []

# ---- WordNet i bag_of_words, tokenize, word frequency ----
data['Timestamp'] = pd.to_datetime(data['Date'])

data['Sentiment'] = data['Tweet'].apply(lambda x: analyze_sentiment(x) if pd.notnull(x) else None)

# while start_index < len(data):
#     subset = data.iloc[start_index:start_index + posts_per_step]
#     subset_list.append(subset)
#     start_index += step_size
#
# merged_data = pd.concat(subset_list, ignore_index=True)
all_tokens = []
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if
              token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)


data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else "")
# i = 0
# for tweet in data["Tweet"]:
#     if pd.notnull(tweet):  # Sprawdź, czy wartość nie jest NaN
#         tokens = word_tokenize(str(tweet))  # Przekształć wartość na ciąg znaków
#         lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
#         all_tokens.extend(lemmatized_tokens)
vectorizer = CountVectorizer(max_features=50)
X = vectorizer.fit_transform(data['Clean_Tweet'])
df_bag_of_words = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Bag of Words Representation:")
print(df_bag_of_words)

# stop_words = set(stopwords.words('english'))
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
# data['Date'] = pd.to_datetime(data['Date'])
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


# ---- WordNet i bag_of_words, tokenize, word frequency ----


# --- grafy ---

# stop_words = set(stopwords.words('english'))
#
# lemmatizer = WordNetLemmatizer()
# additional_stop_words = ["also", '’', '\"', '“', '”', "v", "vs"]
# stop_words.update(additional_stop_words)
# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
#     return tokens
#
# data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else [])
#
# # Wyodrębnienie najczęściej występujących słów
# all_words = [word for tokens in data['Clean_Tweet'] for word in tokens]
# word_freq = Counter(all_words)
# common_words = [word for word, freq in word_freq.most_common(40)]  # Najczęściej występujące 50 słów
#
# # Tworzenie macierzy współwystępowania
# cooccurrence = Counter()
#
# for tokens in data['Clean_Tweet']:
#     for pair in combinations(set(tokens), 2):
#         if pair[0] in common_words and pair[1] in common_words:
#             cooccurrence[pair] += 1
#
# # Przekształcenie macierzy współwystępowania w DataFrame
# edges = pd.DataFrame(cooccurrence.items(), columns=['pair', 'count'])
# edges[['word1', 'word2']] = pd.DataFrame(edges['pair'].tolist(), index=edges.index)
# edges = edges.drop(columns=['pair'])
#
# # Tworzenie grafu
# G = nx.Graph()
#
# # Dodawanie krawędzi do grafu
# for index, row in edges.iterrows():
#     G.add_edge(row['word1'], row['word2'], weight=row['count'])
#
# # Ustalanie pozycji węzłów
# pos = nx.spring_layout(G, k=0.4, iterations=100)  # k to skala repulsji między węzłami
#
# # Wizualizacja grafu
# plt.figure(figsize=(16, 16))
#
# # Ustalanie rozmiarów węzłów na podstawie częstości występowania słów
# node_sizes = [word_freq[word] * 0.1 for word in G.nodes()]
#
# # Rysowanie węzłów
# nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9)
#
# # Rysowanie krawędzi
# edge_weights = [d['weight']*1 for (u, v, d) in G.edges(data=True)]
#
# scaler = MinMaxScaler(feature_range=(1, 10))
# weights_normalized = scaler.fit_transform([[weight] for weight in edge_weights]).flatten()
# for (u, v), weight in zip(G.edges(), weights_normalized):
#     G[u][v]['weight'] = weight
# nx.draw_networkx_edges(G, pos, width=weights_normalized, alpha=0.7)
#
# # Rysowanie etykiet
# nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
#
# plt.title('Graf współwystępowania najczęściej występujących słów w tweetach')
# plt.axis('off')
# plt.show()

# --- klastyeryzacja ----

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
# def preprocess(text):
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
#     return ' '.join(tokens)
#
# data['Clean_Tweet'] = data['Tweet'].apply(lambda x: preprocess(x) if pd.notnull(x) else '')

#
# vectorizer = TfidfVectorizer(max_features=1000)
# X = vectorizer.fit_transform(data['Clean_Tweet'])
# lda = LatentDirichletAllocation(n_components=10, random_state=0)
# lda.fit(X)
#
# def display_topics(model, feature_names, no_top_words):
#     arr = []
#     for topic_idx, topic in enumerate(model.components_):
#         print(f"Topic {topic_idx}:")
#         print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
#         arr.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
#     return arr
#
#
# array_topics = display_topics(lda, vectorizer.get_feature_names_out(), 3)
# print(array_topics)
# kmeans = KMeans(n_clusters=10, random_state=0)
# data['Cluster'] = kmeans.fit_predict(X)
#
# # Agregowanie danych dziennie
# data['Date'] = data['Timestamp'].dt.date
# daily_sentiment = data.groupby(['Date', 'Cluster'])['Sentiment'].mean().unstack()
#
# # Wizualizacja wyników
# plt.figure(figsize=(12, 6))
# i = 0
# for cluster in daily_sentiment.columns:
#     plt.plot(daily_sentiment.index, daily_sentiment[cluster], marker='o', label=f'Cluster: {array_topics[i]}')
#     i+=1
# plt.title('Zmiana sentymentu tweetów w czasie dla klastrów')
# plt.xlabel('Data')
# plt.ylabel('Średni sentyment')
# plt.xticks(rotation=45)
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# ----- WordCloud ------

# wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
#
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


"""
Karne chorwacja vs rosja

"""
