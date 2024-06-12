import torch
from torch import nn
import math

# based on https://github.com/brandokoch/attention-is-all-you-need-paper/tree/master and pytorch tutorial

# TODO: look up xavier initialization for weights
# TODO: add start token 

# Transformer model
class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, device, n_heads=1, n_blocks=6, d_model=512, d_ffn=2048, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.flatten = nn.Flatten()                              # flatten the input sequence since we are dealing with images
        self.in_embedding = nn.Embedding(in_dim, d_model)               # input embedding layer
        self.out_embedding = nn.Embedding(out_dim, d_model)             # output embedding layer
        self.in_positional_encoding = Positional_Encoding(d_model, device)      # input positional encoding for encoder
        self.out_positional_encoding = Positional_Encoding(d_model, device)     # output positional encoding for decoder

        self.linear = nn.Linear(d_model, out_dim)                       # linear layer to get output classes
        self.softmax = nn.Softmax(dim=-1)                               # softmax to get probabilities of each class
        
        self.encoder = Encoder(d_model, d_ffn, n_heads, n_blocks, dropout_rate, device)                      # Note: when creating new tensors, need to specify device they are created on (go on cpu by default)
        self.decoder = Decoder(d_model, d_ffn, n_heads, n_blocks, dropout_rate, device)
    
    # TODO: ISSUE - passing in the target is better for sequence generation (language translation). For classification, model starts using the target for prediction directly and doesn't actually "learn" the weights

    def forward(self, x, target):
        x = self.flatten(x).to(dtype=torch.long, device=x.device)                 # converting into another tensor type moves tensor to cpu by default.         
        encoder_in = self.in_embedding(x) * math.sqrt(self.d_model)               # multiply embeddings by sqrt(d_model) as in paper
        encoder_in = self.in_positional_encoding(encoder_in)                      # encoder_in is a (batch_size, seq_len, d_model) tensor
        encoder_output = self.encoder(encoder_in)

        target = target.unsqueeze(1)  
        decoder_in = self.out_embedding(target) * math.sqrt(self.d_model)           # decoder embeds target sequence
        decoder_in = self.out_positional_encoding(decoder_in)
        decoder_output = self.decoder(encoder_output, decoder_in)                   # output dimensions are: (batch size, target sequence length, d_model)
        decoder_output = self.linear(decoder_output)                                # do final linear layer to get output to desired number of classes/sequence length
        return decoder_output                                                                        

# list of all encoder blocks
class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, n_blocks, dropout_rate, device):
        super().__init__()

        self.encoder_layers = nn.ModuleList([Encoder_Block(d_model, d_ffn, n_heads, dropout_rate, device) for _ in range(n_blocks)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)                                    # call on individual encoder block one at a time.
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, n_blocks, dropout_rate, device):
        super().__init__()

        self.decoder_layers = nn.ModuleList(Decoder_Block(d_model, d_ffn, n_heads, dropout_rate, device) for _ in range(n_blocks))

    def forward(self, encoder_output, x):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(encoder_output, x)                        # Transformer actually looks like a U-Net and Encoders are processed first, outputs are saved, and passed into respective decoders.                       
        return x

# list of all decoder blocks
class Encoder_Block(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, dropout_rate, device):
        super().__init__()
        d_model = d_model
        d_ffn = d_ffn                                    # feed forward network layer ~ 4 times the size of d_model
        dropout_rate = dropout_rate

        self.Wq = nn.Linear(d_model, d_model)                    # paper says it uses dim = 512 for outputs for all embeddings
        self.Wk = nn.Linear(d_model, d_model)                    # if implementing multiheaded attention later, good to have these here to use in both mutltiheaded and self-attention classes
        self.Wv = nn.Linear(d_model, d_model)
        self.attention = Multi_Headed_Attention(n_heads, d_model, device)    
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

class Decoder_Block(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, dropout_rate, device):
        super().__init__()
        d_model = d_model
        d_ffn = d_ffn                                              # feed forward network layer ~ 4 times the size of d_model
        dropout_rate = dropout_rate

        # masked self-attention
        self.Wq_1 = nn.Linear(d_model, d_model)                    # paper says it uses dim = 512 for outputs for all embeddings
        self.Wk_1 = nn.Linear(d_model, d_model)                    # if implementing multiheaded attention later, good to have these here to use in both mutltiheaded and self-attention classes
        self.Wv_1 = nn.Linear(d_model, d_model)
        self.attention1 = Multi_Headed_Attention(n_heads, d_model, device)    
        self.dropout1 = nn.Dropout(dropout_rate)
        self.add_and_norm1 = Add_and_Norm(d_model)

        # encoder-decoder attention (using keys and values from encoder)
        self.Wq_2 = nn.Linear(d_model, d_model)
        self.Wk_2 = nn.Linear(d_model, d_model)
        self.Wv_2 = nn.Linear(d_model, d_model)
        self.attention2 = Multi_Headed_Attention(n_heads, d_model, device)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.add_and_norm2 = Add_and_Norm(d_model)                   

        # feed forward network
        self.ffn = Position_wise_ffn(d_model, d_ffn)            
        self.dropout3 = nn.Dropout(dropout_rate)
        self.add_and_norm3 = Add_and_Norm(d_model)

    def forward(self, encoder_output, x):                   # x = decoder input, which is composed of the target sequence embeddings
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

# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=class%20PositionalEncoding(nn.Module)%3A 
class Positional_Encoding(nn.Module):                    
    def __init__(self, d_model, device, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoding = torch.zeros([1, max_len, d_model], device=device)                       # each "word" has encoding of size d_model

        # calculate e^(2i * log(n)/d_model) where n = 10000 from original paper and i goes from 0 to d_model/2 because there are d_model PAIRS
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(torch.tensor(10000.0)) / d_model))  

        # create (max_len, 1) column tensor for all positions (numbered)
        pos = torch.arange(0, max_len).unsqueeze(1)

        # broadcast and set even indices to sin  and odd indices to cos
        self.pos_encoding[0, :, 0::2] = torch.sin(pos * div_term)                  # select all rows. Start at column 0 and skip every 2 cols
        self.pos_encoding[0, :, 1::2] = torch.cos(pos * div_term)                  

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1)]                                   # trim down pos_encoding to size of actual input sequence. dim = (1, seq_len, d_model)
        x = self.dropout(x)                          
        return x

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

class Multi_Headed_Attention(nn.Module):
    def __init__(self, n_heads, d_model, device):
        super().__init__()
        self.n_heads = n_heads

        # If doesn't divide evenly, concatenation won't work.
        assert d_model % n_heads == 0
        self.d_key = d_model // n_heads

        self.attention = Self_Attention(device)

    def forward(self, q, k, v, is_masked=False):
        # 1 - split into heads
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_key)                # (batch size, sequence len, num heads, d_key)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_key)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_key)

        # 2 - swap n_heads and sequence length dimensions to split each batch along each head
        q = q.transpose(1, 2)                                                       # (batch size, n_heads, sequence len, d_key)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3 - pass into self-attention layer
        output = self.attention(q, k, v, is_masked)

        # 4 - Reverse step 2 then concatenate heads
        output = output.transpose(1, 2).contiguous()                                             # Need contiguous because .view() changes the way the tensor is stored (not stored consecutively in memory anymore)
        output = output.view(output.size(0), -1, self.d_key * self.n_heads)
        return output


class Self_Attention(nn.Module):            # q and k have dimensions d_v by d_k
    def __init__(self, device):
        super().__init__()  
        self.device = device      

    def forward(self, q, k, v, is_masked, padding=0):
        d_k = q.size(-1)                                                                                   # get last dimension of q (should be d_k)
        padding = padding                                                                            
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)                          # want last two dimensions to get swapped
        
        # source/padding mask. Safer to make this into a boolean mask rather than rounding so that numbers won't accidentally be rounded to 0
        mask = attention_weights != padding                                                                # make sure padding values are not considered in softmax

        # target mask (for decoder)
        if is_masked:                       
            # combine padding and target mask. Should have dimensions (target_sequence_len, target_sequence_len)
            mask = torch.tril(torch.ones([1, attention_weights.size(-1), attention_weights.size(-1)], device=self.device)).bool() & mask    # target sequence length for dim = -1 should be the same as dim = -2   
        
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)                                  # set all values we want to ignore to -infinity
        
        probabilities = torch.softmax(attention_weights, dim=-1)                                            # gets the probabilities along last dimension. For 2d the result of softmax is a (d_v, 1) vector.
        return torch.matmul(probabilities, v)      
    

class Add_and_Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, output):
        return self.norm(input + output)