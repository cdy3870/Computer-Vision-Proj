from __future__ import print_function
import torch
#import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec, KeyedVectors
#import torch.optim as optim

"""class word2vec(nn.Module):
    def __init__(self, vocabulary_size, embedding_dims):
        super(word2vec, self).__init__()
        
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dims)
        # self.i2h = nn.Linear(vocabulary_size, embedding_dims)
        self.l1 = nn.Linear(embedding_dims, 32)
        self.l2 = nn.Linear(32, vocabulary_size)
        self.softmax = nn.LogSoftmax(dim=1)#(dim=1) #its not Softmax
        self.dropout = nn.Dropout(p= 0.1)
        
    def forward(self, input):

        # print('\n', 'input.shape: ', input.shape)
        
        embeddings = self.embeddings(input)
        # Input: LongTensor (N, W), N = mini-batch, W = number of indices to extract per mini-batch
        # Output: (N, W, embedding_dim)
        # print('embeddings.shape: ', embeddings.shape)
        # exit()
        embeddings = embeddings.squeeze()
        # print('embeddings.shape: ', embeddings.shape) #(70, 5)
        
        # exit()
        out1 = self.dropout(self.l1(F.relu(embeddings)))                                     #ReLU here too ???
        # print('out1.shape: ', out1.shape) #(70, 15)
        out2 = self.dropout(self.l2(F.relu(out1)))
        # print('out2.shape: ', out2.shape)
        log_probs = self.softmax(out2)
        # print('log_probs.shape: ', log_probs.shape) #(70, 15)
        # exit()
        return out2, log_probs
    
    def predict(self, test_input):
        word_embedding = self.i2h(test_input)
        return word_embedding"""

class Word2Vec():
    def __init__(self, model_url):
        self.model = KeyedVectors.load_word2vec_format(model_url, binary=True, unicode_errors='replace')

    def get_embedding_dims(self):
        return self.model.vector_size



def main():
    #vocabulary_size = 
    #embedding_dims = 50
    model = Word2Vec("word2vec_model.txt").get_model()
    
    print(model['bird'])


if __name__ == "__main__":
    main()
