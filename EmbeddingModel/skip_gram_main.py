class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipgramModeler, self).__init__()
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, context_size*vocab_size)
    
    def forward(self,x):
        embeds = self.embeddings(x).view(1,-1)
        output = self.linear1(embeds)
        output = F.relu(output)
        output = self.linear2(output)
        log_probs = F.log_softmax(output, dim=1).view(self.context_size, -1)
        return log_probs