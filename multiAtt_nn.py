import torch
from torch import nn, optim
import math

class MultiAtt_NN(nn.Module):
    """
    Softmax classifier for sentence-level relation extraction.
    """
    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.output_embedding = nn.Embedding(num_class,self.sentence_encoder.hidden_size)
        self.output_embedding.weight.data.copy_(torch.randn(num_class, self.sentence_encoder.hidden_size))
        # r = math.sqrt(6/(num_class+self.sentence_encoder.hidden_size))
        # self.output_embedding.weight.data.copy_(torch.rand(num_class,self.sentence_encoder.hidden_size)*(-2*r)+r)
        self.U = nn.Parameter(torch.randn(self.sentence_encoder.hidden_size,num_class))
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool1d(self.sentence_encoder.max_length,stride=1)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        _item = self.sentence_encoder.tokenize(item)
        item = []
        for x in _item:
            item.append(x.to(next(self.parameters()).device))
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            W, (B, H)
        """
        R_star = self.sentence_encoder(*args) # (B, H)
        W = self.pooling_attention(R_star)
        return W

    def pooling_attention(self,R_star):
        RU = torch.matmul(R_star.transpose(1,2),self.U)
        G = torch.matmul(RU,self.output_embedding.weight.data)
        AP = self.softmax(G)
        W = self.maxpool(torch.mul(R_star,AP.transpose(1,2))).squeeze(-1)
        return W

    def logit_to_score(self, logits):
        return torch.softmax(logits, -1)