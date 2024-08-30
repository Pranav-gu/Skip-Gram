import numpy as np
import pandas as pd
import torch
import re

data = pd.read_csv('train.csv')

import warnings
warnings.filterwarnings('ignore')
class SkipGramWordEmbedding():
    def __init__(self, learning_rate, window_size, num_epochs, embedding_size=300, negative_samples=5):
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.word_index = {}
        self.index_word = {}
        self.word_embeddings = {}
        self.context_embeddings = {}
        self.probabilities = {}
        self.word_appear = {}
        self.total_words = 0
    


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    


    def backprop(self, positive_samples, negative_samples_indices, word):
        # Positive samples
        pos_context_embeddings = np.array([self.context_embeddings[word] for word in positive_samples])
        pos_dot = np.dot(pos_context_embeddings, self.word_embeddings[word])
        pos_sigmoid = self.sigmoid(pos_dot)
        pos_samples_grad = np.zeros((pos_sigmoid.shape[0], self.word_embeddings[word].shape[0]), dtype = np.float32)
        word_embedding_grad_pos = np.zeros((pos_sigmoid.shape[0], self.word_embeddings[word].shape[0]), dtype = np.float32)
        for i in range(pos_sigmoid.shape[0]):
            pos_samples_grad[i] = (pos_sigmoid[i] - 1) * self.word_embeddings[word]
            word_embedding_grad_pos[i] = (pos_sigmoid[i] - 1) * pos_context_embeddings[i]


        # Negative samples
        neg_context_embeddings = np.array([self.context_embeddings[self.index_word[index]] for index in negative_samples_indices])
        neg_dot = np.dot(neg_context_embeddings, self.word_embeddings[word])
        neg_sigmoid = self.sigmoid(neg_dot)

        neg_samples_grad = np.zeros((neg_sigmoid.shape[0], self.word_embeddings[word].shape[0]), dtype = np.float32)
        word_embedding_grad_neg = np.zeros((neg_sigmoid.shape[0], self.word_embeddings[word].shape[0]), dtype = np.float32)
        for i in range(neg_sigmoid.shape[0]):
            neg_samples_grad[i] = (neg_sigmoid[i] - 1) * self.word_embeddings[word]
            word_embedding_grad_neg[i] = (neg_sigmoid[i] - 1) * neg_context_embeddings[i]

        
        # Gradient update
        for i, sample in enumerate(positive_samples):
            self.context_embeddings[sample] -= self.learning_rate*pos_samples_grad[i]
        for i, sample in enumerate(negative_samples_indices):
            self.context_embeddings[self.index_word[sample]] -= self.learning_rate*neg_samples_grad[i]
        self.word_embeddings[word] -= self.learning_rate * (np.sum(word_embedding_grad_pos, axis=0) + np.sum(word_embedding_grad_neg, axis=0))

        loss = -(np.sum(np.log(pos_sigmoid))+np.sum(np.log(neg_sigmoid)))
        return loss


    def skip_gram(self, sentences):
        print("Training Skip-Gram Classifier")
        for epoch in range(self.num_epochs):
            loss = 0
            samples = 0
            for sentence in sentences:
                for i, word in enumerate(sentence):
                    positive_samples = sentence[max(0, i - self.window_size):i] + sentence[i + 1:min(i + 1 + self.window_size, len(sentence))]
                    negative_samples_indices = np.random.randint(0, len(self.index_word), size=self.negative_samples)
                    loss += self.backprop(positive_samples, negative_samples_indices, word)
                    samples += 1
        return



    def fit(self, corpus):
        print("Data Preprocessing")
        index = 0
        sentences = []
        for i in range(len(corpus)):
            sentence = re.findall(r"[\w']+|[.,!?;'-]", corpus[i])
            for word in sentence:
                self.total_words += 1
                if word not in self.word_index:
                    embeddings = np.random.randn(self.embedding_size).astype(np.float32)
                    embeddings1 = np.random.randn(self.embedding_size).astype(np.float32)
                    self.word_embeddings[word] = embeddings
                    self.context_embeddings[word] = embeddings1
                    self.word_index[word] = index
                    self.index_word[index] = word
                    self.probabilities[word] = 0
                    index += 1
                self.probabilities[word] += 1
                self.word_appear[self.total_words - 1] = self.word_index[word]
            sentences.append(sentence)

        for word in self.probabilities:
            self.probabilities[word] /= self.total_words
        self.skip_gram(sentences)
        return
    

    def most_similar(self, word, most_freq_num):
        l = []
        if word not in self.word_embeddings:
            return l

        distances = np.zeros((len(self.word_embeddings)), dtype = np.float32)
        for i, sample in enumerate(self.word_embeddings):
            if (word == sample):
                continue
            distances[i] = np.dot(self.word_embeddings[sample], self.word_embeddings[word])
        
        curr_index = self.word_index[word]
        most_similar_indices = np.argsort(distances)[:most_freq_num+1]
        for i in most_similar_indices:
            if i != curr_index:
                l.append((self.index_word[i], distances[i]))
        return l


skip_gram_embeddings = SkipGramWordEmbedding(learning_rate=0.0001, window_size=2, num_epochs = 5)
skip_gram_embeddings.fit(data['Description'])
similar_words = skip_gram_embeddings.most_similar(word = "Reuters", most_freq_num = 5)               # pass a word as an input
for i, word in enumerate(similar_words):
    print(f"{i}\t-->\t{word}")


for key in skip_gram_embeddings.word_embeddings:
    skip_gram_embeddings.word_embeddings[key] = (skip_gram_embeddings.word_embeddings[key]+skip_gram_embeddings.context_embeddings[key])/2
torch.save(skip_gram_embeddings.word_embeddings, "skip-gram-word-vectors.pt")