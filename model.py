import torch
from torch import nn
import math

# based on https://github.com/brandokoch/attention-is-all-you-need-paper/tree/master and pytorch tutorial

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, device, n_heads=8, n_blocks=6, d_model=512, d_ffn=2048, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = Positional_Encoding(in_dim, d_model, device)         # input positional encoding for encoder

        # MLP head from ViT
        self.mlp_head = nn.Linear(d_model, out_dim)                               # linear layer to get output classes
        self.tanh = nn.Tanh()                                                   # non linearity activation function used in MLP head

        self.encoder = Encoder(d_model, d_ffn, n_heads, n_blocks, dropout_rate, device)
        self.apply(self.init_weights)

    # Initialize weights to very small numbers close to 0, instead of pytorch's default initalization. 
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.001)
        if isinstance(m, nn.Conv2d):
            pass                # TODO: do I need to initialize patch embedding?
            #torch.nn.
    
    def forward(self, x, target):                                                       
        encoder_in = self.positional_encoding(x)                       # encoder_in is a (batch_size, seq_len, d_model) tensor
        encoder_output = self.encoder(encoder_in)                      # output = (batch_size, num_tokens, d_model)

        # Take out class token and run MLP head only on class token
        class_token_learned = encoder_output[:, 0, :]
        class_token_learned = self.tanh(class_token_learned)
        output = self.mlp_head(class_token_learned)     
        return output
    
# list of all encoder blocks
class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, n_heads, n_blocks, dropout_rate, device):
        super().__init__()

        self.encoder_layers = nn.ModuleList([Encoder_Block(d_model, d_ffn, n_heads, dropout_rate, device) for _ in range(n_blocks)])

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)                                    # call on individual encoder block one at a time.
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
    

### Helper classes ###

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


# separate image into patches and do positional embedding + patch embedding for ViT
# Followed: 
class Patch_Embedding(nn.Module):
    def __init__(self, image_size, patch_size, n_channels, d_model, device, dropout_rate=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model          # same as embedding dimension 
        self.num_patches = (image_size // patch_size) ** 2 + 1          # add 1 for the class token
        
        self.conv = torch.nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)         # no overlapping means stride needs to be same as patch size
        self.pos_encoding = nn.Parameter(torch.zeros([1, self.num_patches, d_model], device=device))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)        # (batch size, channels, height, width) => (batch size, d_model, height/patch size, width/patch size)
        x = x.flatten(2)         # result = (batch size, d_model, num_patches). Look at num_patch definition above for the math
        x = x.transpose(1, 2)           # switch to having num_patch tensors of dimension d_model (512)
        return x


# Followed: https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=class%20PositionalEncoding(nn.Module)%3A 
class Positional_Encoding(nn.Module):                     
    def __init__(self, in_dim, d_model, device, dropout_rate=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten() 
        self.in_embedding = nn.Embedding(in_dim, d_model) 
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model)) 
        self.pos_encoding = torch.zeros([1, max_len, d_model], device=device)                       # each "word" has encoding of size d_model

        # calculate e^(2i * log(n)/d_model) where n = 10000 from original paper and i goes from 0 to d_model/2 because there are d_model PAIRS
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(torch.tensor(10000.0)) / d_model))  

        # create (max_len, 1) column tensor for all positions (numbered)
        pos = torch.arange(0, max_len).unsqueeze(1)

        # broadcast and set even indices to sin and odd indices to cos
        self.pos_encoding[0, :, 0::2] = torch.sin(pos * div_term)                  # select all rows. Start at column 0 and skip every 2 cols
        self.pos_encoding[0, :, 1::2] = torch.cos(pos * div_term)    

        # make positional embedding a parameter so it can be learned
        self.pos_encoding = nn.Parameter(self.pos_encoding)             

    def forward(self, x):
        x = self.flatten(x).to(dtype=torch.long, device=x.device)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        embedded_img = self.in_embedding(x) * math.sqrt(self.d_model)
        x = torch.cat((class_token, embedded_img), dim=1)               # concatenate the class token with the flattened image embedding along num_tokens dimension (dim = 1)

        x = x + self.pos_encoding[:, :x.size(1)]                                   # trim down pos_encoding to size of actual input sequence. dim = (1, num_tokens, d_model)
        x = self.dropout(x)                          
        return x

# called "position wise" because it already includes positional embeddings from previous layers
class Position_wise_ffn(nn.Module):                           # 2 fully connected dense layers  https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c 
    def __init__(self, d_model, d_ffn):                       # feed forward just means no recurrent relations
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)         
        x = self.gelu(x)
        x = self.linear2(x)
        return x    

class Add_and_Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, output):
        return self.norm(input + output)