import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
#%% --------------- FONCTIONS ---------------
#%% --------------- 1/ Chargement des données ---------------
data = pd.read_csv('QueryResults_Score500.csv', sep=',')
#%% --------------- 2/ Nettoyage des données --------------- 
# Pensez à inclure une fréquence des mots utilisés
data_v1 = data.dropna().drop(['Body'], axis = 1)
#%% On va tokenized les Tiles et les Tags
data_v1['tokenized_sentence_title'] =  data_v1.apply(lambda row: nltk.word_tokenize(row['title']), axis = 1)
data_v1['tokenized_sentence_Tags'] =  data_v1.apply(lambda row: nltk.word_tokenize(row['Tags']), axis = 1)
#%%
data_v1['tokenized_sentence'] = data_v1['tokenized_sentence_title'] + data_v1['tokenized_sentence_Tags']
#%% Réalisons la stemmatisation (+ MAJUSCULE -> minuscule), nous utiliserons count_vectorizer afin d'enlever les stopwords
pt = PorterStemmer()
data_v1['tokenized_stemmed_sentence'] =  data_v1.apply(lambda row: [pt.stem(w) for w in row['tokenized_sentence']], axis = 1)
#Indiquer j'ai testé mais pas de bons résultats / je préfère ne pas l'utiliser
#%% Utisons WordVectorizer qui va nous permettre 
# Virer les numériques / regarder les cas '-' / '.' / '. ' 
data_v1['stemmed_sentence'] =  data_v1.apply(lambda row: ' '.join(row['tokenized_stemmed_sentence']), axis = 1)
stemmed_sentence = data_v1['stemmed_sentence'].tolist()
CV = CountVectorizer(token_pattern = '[A-Za-z]+(?=\\s+)', lowercase = True, stop_words = 'english', max_df = 500, min_df = 3)
SS1 = CV.fit_transform(stemmed_sentence)
CV.vocabulary_
CV.stop_words_
vocab = CV.get_feature_names()
#%% --------------- 3/ Utilisation de LDA --------------- 
LDA = LatentDirichletAllocation(n_components = 25)
id_topic = LDA.fit_transform(SS1)
#%% 
n_top_words = 10
topic_words = {}

for topic, comp in enumerate(LDA.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [vocab[i] for i in word_idx]

for topic, words in topic_words.items():
    print('Topic: %d' % topic)
    print('  %s' % ', '.join(words))
#%% --------------- 4/ Exemple --------------- 
test = ['How to use javascript and html ?']
test1 = CV.transform(test)
test1_ = LDA.transform(test1)
