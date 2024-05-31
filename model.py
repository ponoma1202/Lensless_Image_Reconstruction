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
        self.dropout1 = nn.Dropout(0.1)
        self.add_and_norm1 = Add_and_Norm(d_model)      # TODO: should I implement Add_and_Norm class separately?

        self.ffn = Position_wise_ffn(d_model, d_ffn)    # output has dimension d_model
        self.dropout2 = nn.Dropout(0.1)
        self.add_and_norm2 = Add_and_Norm(d_model)

    def forward(self, q, k, v, x):
        Q = self.q(q)
        K = self.k(k)
        V = self.v(v)
        attention = attention(Q, K, V)
        self.dropout1(attention) 
        x = self.add_and_norm1(attention, x)    # x is original input embedding

        ffn = self.ffn(x)
        ffn = self.dropout2(ffn)
        ffn = self.add_and_norm2(x, ffn)
        return ffn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


# Helper classes

class Position_wise_ffn(nn.Module):         # 2 fully connected dense layers  https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c 
    def __init__(self, d_model, d_ffn):     # feed forward just means no recurrent relations
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
    
    def forward(self, x):
        x = self.linear1(x)         
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class Self_Attention(nn.Module):            # q and k have dimensions d_v by d_k
    def __init__(self):
        super().__init__()          # TODO: implement mask option

    def forward(self, q, k, v):
        d_k = q.size(-1)                    # get last dimension of q (should be d_k)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)      # want last two dimensions to get swapped
        probabilities = torch.softmax(attention_weights, dim=-1)  # gets the probabilities along last dimension. For 2d the result of softmax is a d_v by 1 vector.
        return torch.matmul(probabilities, v)  # multiply probabilities with v to get the weighted sum of v. 
    

class Add_and_Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, output):
        return self.norm(input + output)