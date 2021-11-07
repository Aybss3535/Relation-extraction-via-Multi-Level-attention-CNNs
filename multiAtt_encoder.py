from base_encoder import BaseEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch

class MultiAttEnocder(BaseEncoder):
    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 position_size=5,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3,
                 padding_size=1,
                 dropout=0,
                 activation_function=torch.tanh,
                 mask_entity=False):
        super(MultiAttEnocder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, blank_padding, word2vec, mask_entity=mask_entity)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        self.softmax = nn.Softmax(dim=0)
        self.minus = -1e10
        self.conv = nn.Conv1d(self.input_size, self.hidden_size, self.kernel_size, padding=self.padding_size,bias=True)
        self.pool = nn.MaxPool1d(self.max_length)

    def forward(self, token, pos1, pos2,e1,e2):
        token_embedding = self.word_embedding(token) #(B,L,word_EMBED)
        x = torch.cat([token_embedding,
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)], 2)  # (B, L, EMBED)
        R_star = self.input_attention(e1,e2,token_embedding,x) #(H,B,L)
        # R_star = self.act(self.conv(R.transpose(1,2)))
        return R_star

    def input_attention(self,e1,e2,token_embedding,x):
        e1_embedding = self.word_embedding(e1.squeeze(-1))  # (B,word_EMBED)
        e2_embedding = self.word_embedding(e2.squeeze(-1))
        alpha_1 = self.softmax(torch.bmm(token_embedding,e1_embedding.unsqueeze(-1)))
        alpha_2 = self.softmax(torch.bmm(token_embedding,e2_embedding.unsqueeze(-1)))
        alpha = (alpha_1 + alpha_2) / 2  # (L,B)
        R = self.act(self.conv(x.transpose(1,2)))
        R_star = torch.mul(R,alpha.transpose(1,2))
        return R_star

    def tokenize(self, item):
        return super().tokenize(item)
