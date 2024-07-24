import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(num_embeddings=vocab_size, embedding_dim=d_model)


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, zeros_pad=True, scale=True):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.vocab_size, self.d_model, self.zeros_pad = vocab_size, d_model, zeros_pad
        self.zeros_pad, self.scale = zeros_pad, scale
        self.embedding_table = Parameter(torch.Tensor(vocab_size, d_model))
        nn.init.xavier_normal_(self.embedding_table.data)
        if self.zeros_pad:
            self.embedding_table.data[0, :].fill_(0)

    def forward(self, x):
        self.padding_idx = 0 if self.zeros_pad else -1
        outputs = F.embedding(x, self.embedding_table, 0, None, 2, False, False)
        if self.scale:
            outputs = outputs * (self.d_model ** 0.5)
        return outputs


class PositionalEmbedding(nn.Module):
    """
    位置编码模块的目标是生成一个位置编码的矩阵，其形状为 max_len * d_model，
    其中 max_len 表示模型允许的最大输入序列长度（单词数量），d_model 表示
    Embedding 的维度。位置编码表中每一行表示一个位置（1 ~ max_len）对应的
    位置编码。在 Transformer 中，该位置编码将被加和到原始 Embedding 从而
    形成位置编码后的向量。
    """

    def __init__(self, d_model, max_len):
        """
        位置编码公式：
        - PE(pos, 2i) = sin(pos / (10000^(2i / d_model)))
        - PE(pos, 2i+1) = cos(pos / (10000^(2i / d_model)))

        Parameter
        ----------
        d_model: 模型中潜在表示向量（Embedding）的维度
        max_len: 模型允许的最大输入序列长度
        """
        super(PositionalEmbedding, self).__init__()
        embedding_table = torch.zeros(max_len, d_model, dtype=torch.float32, requires_grad=False)
        pos = torch.arange(start=0, end=max_len, dtype=torch.float32).unsqueeze(1)  # pos => dimension: max_len * 1
        _2i = torch.arange(start=0, end=d_model, step=2, dtype=torch.float32)  # _2i => dimension: d_model_2 = d_model / 2
        embedding_table[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))  # _2i => dimension after broadcast: 1 * d_model_2 => max_len * d_model_2
        embedding_table[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.embedding_table = embedding_table

    def forward(self, x):
        """
        输入 row 行 Embedding（每次输入的序列长度可能不一样，不一定
        每次都是 max_len 个），每一行代表一个单词的潜在表示。本方法返
        回位置编码表中的前 row 行作为位置编码。
        """
        row = x.shape[0]
        return self.embedding_table[:row, :]
    
    @classmethod
    def test_position_encoding(cls):
        d_model = 6   # Embedding 向量的维度
        max_len = 10  # 输入序列的最大长度（最大单词数量）
        pe = PositionalEmbedding(d_model, max_len)
        x = torch.randn(10, d_model)  # 随机生成 10 个 embedding 作为输入
        pe_result = pe.forward(x)
        print("---------x----------")
        print(x)
        print("----------position_embedding---------")
        print(pe_result)
        print("----------encoded_x---------")
        print(x + pe_result)


class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob, sinusoid=True, zeros_pad=True, scale=True) -> None:
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len) if sinusoid else AbsolutePositionalEmbedding(vocab_size, d_model, zeros_pad=zeros_pad, scale=scale)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, n_head) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # 1. 获取维度参数
        batch_size, max_len, d_model = q.shape
        dim_per_head = d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. 计算 Q，K，V 和点积注意力分数（多头分开） Q, K, V => [64=batch_size, 8=n_head, 50=max_len, 64=dim_per_head]
        q = q.view(batch_size, max_len, self.n_head, dim_per_head).permute(0, 2, 1, 3)
        k = k.view(batch_size, max_len, self.n_head, dim_per_head).permute(0, 2, 1, 3)
        v = v.view(batch_size, max_len, self.n_head, dim_per_head).permute(0, 2, 1, 3)
        score = q @ k.transpose(2, 3) / math.sqrt(dim_per_head)
        # 3. 为注意力分数增加掩码（Decoder 模块使用）  score => [64, 8, 50, 50]
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score)
        # 4. 计算注意力分数的加权输出
        weighted_v = score @ v  # weighted_v => [64, 8, 50, 64]
        # 5. 连接多头输出
        weighted_v = weighted_v.permute(0, 2, 1, 3).contiguous().view(batch_size, max_len, d_model)  #  weighted_v => [64, 50, 512]
        output = self.w_concat(weighted_v)
        return output

    @classmethod
    def test_attention(cls):
        batch_size = 3
        d_model = 4
        n_head = 2
        attention = MultiHeadSelfAttention(d_model, n_head)
        x = torch.randn(batch_size, 4, d_model)  # 随机生成 3 批，每批 4 个 embedding 作为输入
        output = attention(x)
        print("---------x----------")
        print(x)
        print("-----------multi_head_self_attention_output-------------")
        print(output)
        print(output.shape)


class LayerNorm(nn.Module):

    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + 1e-10)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, ffn_hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)

    def forward(self, x, mask):
        # 1. 计算自注意力
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        # 2. 计算前馈神经网络
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):

    def __init__(self, n_layers, vocab_size, d_model, max_len, n_head, ffn_hidden, drop_prob) -> None:
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob, sinusoid=False, zeros_pad=True, scale=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, drop_prob=drop_prob) for _ in range(n_layers)])
    
    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, ffn_hidden, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadSelfAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.norm1 = LayerNorm(d_model=d_model)
        self.enc_dec_attention = MultiHeadSelfAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
    
    def forward(self, x_encoder, x_decoder, src_mask, trg_mask):
        # 1. 计算自注意力
        _x_decoder = x_decoder
        x_decoder = self.attention1(q=x_decoder, k=x_decoder, v=x_decoder, mask=trg_mask)
        x_decoder = self.dropout1(x_decoder)
        x_decoder = self.norm1(x_decoder + _x_decoder)
        # 2. 计算第二层自注意力，编码器输出作为 K，V
        if x_encoder is not None:
            _x_decoder = x_decoder
            x_decoder = self.enc_dec_attention(q=x_decoder, k=x_encoder, v=x_encoder, mask=src_mask)
            x_decoder = self.dropout2(x_decoder)
            x_decoder = self.norm2(x_decoder + _x_decoder)
        # 3. 计算前馈神经网络
        _x_decoder = x_decoder
        x_decoder = self.ffn(x_decoder)
        x_decoder = self.dropout3(x_decoder)
        x_decoder = self.norm3(x_decoder + _x_decoder)
        return x_decoder


class Decoder(nn.Module):

    def __init__(self, n_layers, vocab_size, d_model, max_len, n_head, ffn_hidden, drop_prob) -> None:
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob, sinusoid=False, zeros_pad=False, scale=False)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden, drop_prob=drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x_encoder, x_decoder, src_mask, trg_mask):
        x_decoder = self.embedding(x_decoder)
        for layer in self.layers:
            x_decoder = layer(x_encoder=x_encoder, x_decoder=x_decoder, src_mask=src_mask, trg_mask=trg_mask)
        output = self.linear(x_decoder)
        return output


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.device = args.device
        self.src_pad_idx = args.src_pad_idx
        self.trg_pad_idx = args.trg_pad_idx
        self.trg_sos_idx = args.trg_sos_idx
        self.encoder = Encoder(n_layers=args.n_layers, vocab_size=args.vocab_size_source, d_model=args.d_model, max_len=args.max_len, n_head=args.n_head, ffn_hidden=args.ffn_hidden, drop_prob=args.drop_prob, sinusoid=args.sinusoid).to(args.device)
        self.decoder = Decoder(n_layers=args.n_layers, vocab_size=args.vocab_size_target, d_model=args.d_model, max_len=args.max_len, n_head=args.n_head, ffn_hidden=args.ffn_hidden, drop_prob=args.drop_prob, sinusoid=args.sinusoid).to(args.device)
    
    def forward(self, src, trg):
        """
        根据原始输入序列和目标输出序列，生成预测的输出序列
        """
        src_mask = self.make_src_mask(src)  # [batch_size, dim_placeholder_multi_head=1, dim_placeholder_d_model=1, d_model]
        trg_mask = self.make_trg_mask(trg)  # [batch_size, dim_placeholder_multi_head=1, d_model, d_model]
        enc_output = self.encoder(x=src, mask=src_mask)
        dec_output = self.decoder(x_encoder=enc_output, x_decoder=trg, src_mask=src_mask, trg_mask=trg_mask)
        return dec_output
    
    def make_src_mask(self, src):
        """
        根据原始输入序列（编码器的输入）生成填充掩码
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        根据目标输出序列（解码器的输入），生成填充掩码和前瞻掩码，并将二者合并作为编码器的掩码矩阵
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
