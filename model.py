import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = 512
        d_ffn = d_model * 4  #feed forward network layer ~ 4 times the size of d_model
        self.q = nn.Linear(d_model, d_model)        # paper says it uses dim = 512 for outputs for all embeddings
        self.k = nn.Linear(d_model, d_model)        # if implementing multiheaded attention later, good to have these here to use in both mutltiheaded and self-attention classes
        self.v = nn.Linear(d_model, d_model)
        self.attention = Self_Attention()     # need to initialize before running forward
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = Position_wise_ffn(d_model, d_ffn)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, x):
        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)
        attention = attention(Q, K, V) 
        attention = self.norm(attention + x)        # x is original input embedding

        ffn = self.ffn(attention)
        ffn = self.norm2(ffn + attention)
        return ffn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


# Helper classes

class Position_wise_ffn(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        x = self.linear1(x)         # TODO add and norm
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class Self_Attention():            # q and k have dimensions d_v by d_k
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        d_k = q.size(-1)                    # get last dimension of q (should be d_k)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)
        probabilities = torch.softmax(attention_weights, dim=-1)  # gets the probabilities along last dimension. For 2d the result of softmax is a d_v by 1 vector.
        return torch.matmul(probabilities, v)  # multiply probabilities with v to get the weighted sum of v. 