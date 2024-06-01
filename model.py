import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, d_model=512, d_ffn=2048, dropout_rate=0.1):
        super().__init__()
        self.encoder_decoder = Encoder_Decoder(in_dim, out_dim, d_model, d_ffn, dropout_rate)
    
    def forward(self, x):
        return self.encoder_decoder(x)

class Encoder_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, d_ffn, dropout_rate):
        super().__init__()
        self.in_embedding = nn.Embedding(in_dim, d_model)               # input embedding layer
        # TODO: outputs are shifted right in the paper????
        self.out_embedding = nn.Embedding(out_dim, d_model)             # output embedding layer
        self.in_positional_encoding = Positional_Encoding(d_model)      # input positional encoding for encoder
        self.out_positional_encoding = Positional_Encoding(d_model)     # output positional encoding for decoder
        self.encoder = Encoder(d_model, d_ffn, dropout_rate)
        self.decoder = Decoder(d_model, d_ffn, dropout_rate)

    def forward(self, x):
        encoder_in = self.in_embedding(x)
        encoder_in = self.in_positional_encoding(encoder_in)
        encoder_output = self.encoder(encoder_in)

        decoder_in = self.out_embedding(x)
        decoder_in = self.out_positional_encoding(decoder_in)
        decoder_output = self.decoder(encoder_output, decoder_in)
        return decoder_output

class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, dropout_rate):
        super().__init__()
        d_model = d_model
        d_ffn = d_ffn                                    # feed forward network layer ~ 4 times the size of d_model
        dropout_rate = dropout_rate

        self.Wq = nn.Linear(d_model, d_model)                    # paper says it uses dim = 512 for outputs for all embeddings
        self.Wk = nn.Linear(d_model, d_model)                    # if implementing multiheaded attention later, good to have these here to use in both mutltiheaded and self-attention classes
        self.Wv = nn.Linear(d_model, d_model)
        self.attention = Self_Attention()    
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_and_norm1 = Add_and_Norm(d_model)     

        self.ffn = Position_wise_ffn(d_model, d_ffn)            # output has dimension d_model
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_and_norm2 = Add_and_Norm(d_model)

    def forward(self, x):
        Q = self.Wq(x)                                  # dimensions = (batch_size, seq_len, d_model) for Q, K, V
        K = self.Wk(x)
        V = self.Wv(x)
        attention_layer = self.attention(Q, K, V)
        attention_layer = self.dropout1(attention_layer) 
        x = self.add_and_norm1(attention_layer, x)            # x is original input embedding

        ffn = self.ffn(x)
        ffn = self.dropout2(ffn)
        ffn = self.add_and_norm2(x, ffn)
        return ffn

class Decoder(nn.Module):
    def __init__(self, d_model, d_ffn, dropout_rate):
        super().__init__()
        d_model = d_model
        d_ffn = d_ffn                                       # feed forward network layer ~ 4 times the size of d_model
        dropout_rate = dropout_rate

        # masked self-attention
        self.Wq_1 = nn.Linear(d_model, d_model)                    # paper says it uses dim = 512 for outputs for all embeddings
        self.Wk_1 = nn.Linear(d_model, d_model)                    # if implementing multiheaded attention later, good to have these here to use in both mutltiheaded and self-attention classes
        self.Wv_1 = nn.Linear(d_model, d_model)
        self.attention1 = Self_Attention()    
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_and_norm1 = Add_and_Norm(d_model)

        # encoder-decoder attention (using keys and values from encoder)
        self.Wq_2 = nn.Linear(d_model, d_model)
        self.Wk_2 = nn.Linear(d_model, d_model)
        self.Wv_2 = nn.Linear(d_model, d_model)
        self.attention2 = Self_Attention()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_and_norm2 = Add_and_Norm(d_model)                   

        # feed forward network
        self.ffn = Position_wise_ffn(d_model, d_ffn)            
        self.dropout3 = nn.Dropout(dropout_rate)
        self.add_and_norm3 = Add_and_Norm(d_model)

    def forward(self, encoder_output, x):
        # masked self-attention
        Q_1 = self.Wq_1(x)
        K_1 = self.Wk_1(x)
        V_1 = self.Wv_1(x)
        attention_layer1 = self.attention1(Q_1, K_1, V_1, is_masked=True)
        attention_layer1 = self.dropout1(attention_layer1)
        x = self.add_and_norm1(attention_layer1, x)

        # encoder-decoder attention
        Q_2 = self.Wq_2(x)
        K_2 = self.Wk_2(encoder_output)
        V_2 = self.Wv_2(encoder_output)
        attention_layer2 = self.attention2(Q_2, K_2, V_2)
        attention_layer2 = self.dropout2(attention_layer2)
        x = self.add_and_norm2(attention_layer2, x)

        # feed forward network
        ffn = self.ffn(x)
        ffn = self.dropout3(ffn)
        ffn = self.add_and_norm3(x, ffn)
        return ffn


# Helper classes
# TODO implement multiheaded attention

class Positional_Encoding(nn.Module):                    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        return 

class Position_wise_ffn(nn.Module):                           # 2 fully connected dense layers  https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c 
    def __init__(self, d_model, d_ffn):                       # feed forward just means no recurrent relations
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
        super().__init__()        

    def forward(self, q, k, v,is_masked=False):
        d_k = q.size(-1)                                                                                    # get last dimension of q (should be d_k)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)                          # want last two dimensions to get swapped
        
        # TODO: what if padding is not 0?
        
        # source/padding mask
        mask = attention_weights.masked_fill(attention_weights == 0, -1e9)                                  # make sure padding values are not considered in softmax by setting them to negative infinity

        # target mask (for decoder)
        if is_masked:                       
            # combine padding and target mask
            mask = torch.tril(torch.ones(1, attention_weights.size(-1), attention_weights.size(-1))) & mask    # target sequence length for dim = -1 should be the same as dim = -2   
        
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)                                  # set all values in upper triangle of matrix to infinity
        
        probabilities = torch.softmax(attention_weights, dim=-1)                                            # gets the probabilities along last dimension. For 2d the result of softmax is a (d_v, 1) vector.
        return torch.matmul(probabilities, v)      
    

class Add_and_Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, output):
        return self.norm(input + output)