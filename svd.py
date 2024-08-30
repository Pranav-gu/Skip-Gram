import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import svds
import re
import torch

data = pd.read_csv('train.csv')

class SVDWordEmbedding():
    def __init__(self, window_size = 2, embedding_size = 300):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.word_index = {}
        self.index_word = {}
        self.embeddings = {}
        self.co_occurrence_matrix = {}
    

    def create_co_occurrence_matrix(self, sentences):
        vocabulary_size = len(self.word_index)
        self.co_occurrence_matrix = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float64)

        for sentence in sentences:
            for i, word in enumerate(sentence):
                curr_index = self.word_index[word]
                context_window_indices = []
                for i in range(max(0, i-self.window_size), i):
                    context_window_indices.append(self.word_index[word])

                for i in range(i+1, min(len(sentence), i+1+self.window_size)):
                    context_window_indices.append(self.word_index[word])

                for context_index in context_window_indices:
                    self.co_occurrence_matrix[curr_index, context_index] += 1
                    self.co_occurrence_matrix[context_index, curr_index] += 1
        return


    def train(self):
        u, sigma, v_t = svds(self.co_occurrence_matrix, k=self.embedding_size)
        self.embeddings = u.dot(np.diag(np.sqrt(sigma)))
        return


    def fit(self, corpus):
        index = 0
        sentences = []
        for i in range(len(corpus)):
            sentence = re.findall(r"[\w']+|[.,!?;'-]", corpus[i])
            for word in sentence:
                if (word not in self.word_index):
                    self.word_index[word] = index
                    self.index_word[index] = word
                    index += 1
            sentences.append(sentence)
        
        self.create_co_occurrence_matrix(sentences)
        self.train()
        return


    def most_similar(self, word, most_freq_num):
        l = []
        if word not in self.word_index:
            return l

        curr_index = self.word_index[word]
        word_vector = self.embeddings[curr_index]
        distances = np.dot(self.embeddings, word_vector)
        most_similar_indices = distances.argsort()[::-1][: most_freq_num+1]
        for i in most_similar_indices:
            if i != curr_index:
                l.append((self.index_word[i], distances[i]))
        return l


svd_embeddings = SVDWordEmbedding()
svd_embeddings.fit(data['Description'])
similar_words = svd_embeddings.most_similar(word = "Reuters", most_freq_num = 5)               # pass a word as an input
for i, word in enumerate(similar_words):
    print(f"{i}\t-->\t{word}")


torch.save(svd_embeddings.embeddings, "svd-word-vectors.pt")
torch.save(svd_embeddings.word_index, "svd-word-vectors-indices.pt")