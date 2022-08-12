import torch as tc
import transformers as trf


def average(emb, mask):
    """averaging a set of embeddings for each instance"""
    if len(emb.shape) == 2:
        emb = emb.unsqueeze(-1)
    assert emb.shape[:2] == mask.shape
    return (mask.unsqueeze(-1) * emb).sum(axis=1) / (5e-7 + mask.sum(axis=-1, keepdims=True))


class SoftSelfAttention(tc.nn.Module):

    """Aggregate a set of instance embeddings by using Soft Self-Attention"""
    
    def __init__(self, dims, mode="first", dropout=0.2, lamda=8.0):
        super(SoftSelfAttention, self).__init__()
        assert mode in ("first", "mean")
        self.first = mode == "first"
        self.scaler = lamda
        self.dropout = tc.nn.Dropout(dropout)
        self.qry_layer = tc.nn.Linear(dims, dims)
        self.key_layer = tc.nn.Linear(dims, dims)

    def forward(self, embs, mask):
        assert len(embs.shape) == 3
        assert len(mask.shape) == 2
        bs, ts, dim = embs.shape

        if self.first:
          qry_emb = embs[:, 0]
          mask[:, 0] = 0.0
        else:
          qry_emb = average(embs, mask)
        qry = self.qry_layer(qry_emb).unsqueeze(1) # (bs, 1, dim)
        key = self.key_layer(embs)    # (bs, ts, dim)
        att = self.scaler * tc.tanh(tc.bmm(qry, key.transpose(2, 1))).squeeze(1)
        att = tc.softmax(self.dropout(att) - (1.0 * mask) * 1e7, -1)
        return average(embs, att)


class GatedFusionExperts(tc.nn.Module):

    """Fuse multiple-resources embeddings with a gated experts structure"""
    
    def __init__(self, in_dims, out_dim, exploration, reduction, dropout):
        super(GatedFusionExperts, self).__init__()
        self.dropout = tc.nn.Dropout(dropout)
        self.out_dim, self.num_exp = out_dim, len(in_dims)
        full_dim = sum(in_dims)
        self.gate1 = SimpleMLP(in_dims[0], in_dims[0] // reduction, in_dims[1], dropout)
        self.gate2 = SimpleMLP(in_dims[1], in_dims[1] // reduction, in_dims[0], dropout)
        self.clf = SimpleMLP(full_dim, full_dim * exploration, out_dim, min(2 * dropout, 0.5))
        
    def forward(self, *X):
        X = [self.dropout(x) for x in X]
        X[0] = X[0] * tc.sigmoid(self.gate2(X[1]))
        X[1] = X[1] * tc.sigmoid(self.gate1(X[0]))
        return self.clf(tc.cat(X, axis=-1))


class BiasRecommender(tc.nn.Module):

    """Predict constant bias to user and items"""
    
    def __init__(self, user_num, item_num):
        super(BiasRecommender, self).__init__()
        self.user_bias = tc.nn.Parameter(tc.ones((1 + user_num, 1)) * 0.1) 
        self.item_bias = tc.nn.Parameter(tc.ones((1 + item_num, 1)) * 0.1)
        self.global_bias = tc.nn.Parameter(tc.FloatTensor([4.0]))

    def forward(self, users, items):
        """
        Inputs
        ------
        users : LongTensor(Batchsize,)
        items : LongTensor(Batchsize,)

        Outputs
        -------
        FloatTensor(Batchsize,)
        """
        assert len(users.shape) == len(items.shape) == 1
        return (self.user_bias[users]  + self.item_bias[items] + self.global_bias).squeeze()


class PLM(tc.nn.Module):

    """Pretrained Language Model"""

    def __init__(self, name, dropout, trainable=False):
        super(PLM, self).__init__()
        self.tokenizer = trf.BertTokenizer.from_pretrained(name, cache_dir="./cache")
        self.encoder = trf.BertModel.from_pretrained(name, cache_dir="./cache")
        self.dropout = tc.nn.Dropout(dropout)
        self.word_dim = self.encoder.config.hidden_size
        self.vocab_size = self.encoder.config.vocab_size
        if trainable is False:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, ids, masks):
        return self.dropout(self.encoder(ids, attention_mask=masks)[0])


class SimpleMLP(tc.nn.Module):
    def __init__(self, in_dim, hide_dim, out_dim, dropout=0.0):
        super(SimpleMLP, self).__init__()
        self.layers = tc.nn.Sequential(
                        tc.nn.Linear(in_dim, hide_dim),
                        tc.nn.LeakyReLU(),
                        tc.nn.Dropout(dropout),
                        tc.nn.Linear(hide_dim, out_dim),
                        )

    def forward(self, x):
        return self.layers(x)
