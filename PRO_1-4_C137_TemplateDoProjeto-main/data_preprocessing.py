# Biblioteca de pré-processamento de dados de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# para stemizar palavras
from nltk.stem import PorterStemmer

# crie um objeto/instância da classe PorterStemmer()
ps = PorterStemmer()

# importando a biblioteca json
import json
import pickle
import numpy as np

words = []  # lista de palavras raízes únicas nos dados
classes = []  # lista de tags únicas nos dados
pattern_word_tags_list = []  # lista do par de (['palavras', 'da', 'frase'], 'tags')

# palavras a serem ignoradas durante a criação do conjunto de dados
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# abrindo o arquivo JSON, lendo os dados dele e o fechando.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# criando função para stemizar palavras
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        # escreva o algoritmo de stemização:
        if word not in ignore_words:
            stemmed_word = ps.stem(word.lower())
            stem_words.append(stemmed_word)
    return stem_words

# criando uma função para criar o corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:
            # tokenize o padrão
            pattern_words = nltk.word_tokenize(pattern)
            # adicione as palavras tokenizadas à lista words
            words.extend(pattern_words)
            # adicione a 'lista de palavras tokenizadas' junto com a 'tag' à lista pattern_word_tags_list
            pattern_word_tags_list.append((pattern_words, intent['tag']))
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    stem_words = get_stem_words(words, ignore_words)
    # Remova palavras duplicadas de stem_words
    stem_words = list(set(stem_words))
    # ordene a lista de palavras-tronco e a lista classes
    stem_words.sort()
    classes.sort()
    # imprima a stem_words
    print('lista de palavras stemizadas: ', stem_words)
    return stem_words, classes, pattern_word_tags_list

# Conjunto de dados de treinamento:
# Texto de Entrada----> como Saco de Palavras (Bag Of Words)
# Tags----------------> como Label
def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        # exemplo: word_tags = (['Ola', 'voce'], 'saudação')
        pattern_words = word_tags[0]  # ['Hi' , 'There]
        bag_of_words = []
        # Stemizando palavras padrão antes de criar o saco de palavras
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)
        # Codificando dados de entrada
        for word in stem_words:
            bag_of_words.append(1) if word in stemmed_pattern_word else bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        # Comece com uma lista de 0s
        labels_encoding = list([0] * len(classes))
        # exemplo: word_tags = (['ola', 'voce'], 'saudação')
        tag = word_tags[1]   # 'saudação'
        tag_index = classes.index(tag)
        # Codificação de etiquetas
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    # Converta as palavras-tronco e a lista classes para o formato de arquivo Python pickle
    with open('stem_words.pkl', 'wb') as f:
        pickle.dump(stem_words, f)
    with open('tag_classes.pkl', 'wb') as f:
        pickle.dump(tag_classes, f)
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    return train_x, train_y

bow_data, label_data = preprocess_train_data()

# depois de completar o código, remova o comentário das instruções de impressão
# print("primeira codificação BOW: " , bow_data[0])
# print("primeira codificação Label: " , label_data[0])
