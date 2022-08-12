import collections
import random

import tqdm
import torch as tc
import numpy as np
import transformers as trf

from .modules import (SoftSelfAttention, GatedFusionExperts,
                     BiasRecommender, SimpleMLP, PLM, average)


class DIRECT(tc.nn.Module):
    
    version = "DIRECT"
    
    def __init__(self, user_num, item_num, aspc_num, aspc_dim, 
                 dropout, device, plm, **args):
        super(DIRECT, self).__init__()
        self._cache = {}
        
        # language modelling
        self.bert = PLM(plm, dropout)
        word_dim = self.bert.word_dim

        # sentiment analysis
        self.sentiment_tagging = SimpleMLP(word_dim, word_dim // 4, 1, min(dropout * 2, 0.5))

        # interest
        self.item_embs = tc.nn.Parameter(tc.zeros((item_num + 1, word_dim)))
        tc.nn.init.xavier_uniform_(self.item_embs)
        self.review_agg = SoftSelfAttention(word_dim, "first", dropout=0.1)
        self.history_agg = SoftSelfAttention(word_dim, "mean", dropout=0.1)
        self.interest_proj = GatedFusionExperts([word_dim, word_dim], aspc_dim,
                                               reduction=8, exploration=4,
                                                dropout=dropout)
        self.interest_norm = tc.nn.BatchNorm1d(aspc_num + 1)
            
        # mention
        self.mention_proj = tc.nn.Linear(word_dim, aspc_dim)

        # aspects
        self.eps, self.threshold = args.get("eps", 0.05) ** 2, args.get("threshold", 0.2)
        self.beta, self.sampling = args["beta"], args["sampling"]
        self.gamma1, self.gamma2, self.gamma3 = args["gamma1"], args["gamma2"], args["gamma3"]
        self.asp_free = tc.tensor([0.] + [1.] * aspc_num).reshape(1, 1, -1).to(device)
        self.asp_eye = tc.eye(aspc_num + 1).to(device)
        self.asp_embs = tc.nn.Parameter(tc.ones((aspc_dim, aspc_num + 1)))
        tc.nn.init.xavier_uniform_(self.asp_embs)
        
        self.bias = BiasRecommender(user_num, item_num)
        self.lossfn = tc.nn.MSELoss() 
        self.device = device
        self.to(device)

    def prepare_item_embedding(self, reviews, batchsize=64):
        reviews = tc.utils.data.DataLoader(reviews, batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)
        with tc.no_grad():
            item_embeddings = [tc.ones((1, self.bert.word_dim))]
            for ids, mask in reviews:
                batch_embs = self.bert.encoder(ids.to(self.device), attention_mask=mask.to(self.device))[0][:, 0]
                item_embeddings.append(batch_embs.cpu())
        self.item_embs = tc.nn.Parameter(tc.cat(item_embeddings, 0).to(self.device), requires_grad=False)
        
    def forward(self, uid, iid, item_doc_ids, item_doc_mask, user_doc_ids, user_doc_mask, user_hist_ids, **args):
        """
        Inputs
        ------
        uid : LongTensor(Batchsize, 1)
        iid : LongTensor(Batchsize, 1)
        item_document_idx : LongTensor(Batchsize, SequenceLength)
        item_document_mask: BinaryTensor(Batchsize, SequenceLength)

        Outputs
        -------
        scores : FloatTensor(Batchsize, 1)
        """
        self._cache.clear()
        uid, iid = uid.to(self.device), iid.to(self.device)
        item_text = item_doc_ids.to(self.device)
        item_mask = item_doc_mask.to(self.device)
        user_hist = user_hist_ids.to(self.device)
        user_text = user_doc_ids.to(self.device)
        user_mask = user_doc_mask.to(self.device)
        
        user_emb = self._encode(user_text, user_mask) 
        item_emb = self._encode(item_text, item_mask)
        sentiment = self._sentiment_analysis(item_emb) # (bs, seq, K)
        gate = (self._mention(item_emb) * self._interest(user_emb, user_mask, user_hist)).sum(axis=-1) * item_mask
        major_score = average(sentiment * gate, gate).squeeze()
        bias_score = self.bias(uid, iid)

        if self.training:
            self._cache["item_ids"] = item_text
            self._cache["item_mask"] = item_mask
            self._cache["user_history"] = user_hist
            self._cache["current_ids"] = args["current_ids"]
            self._cache["current_mask"] = args["current_mask"]
            self._cache["major_score"] = major_score
        return  bias_score + major_score 

    def _encode(self, texts, mask):
        return self.bert(texts, mask) #(bs, seq, dim)

    def _sentiment_analysis(self, embs):
        return tc.tanh(self.sentiment_tagging(embs)).squeeze()  # (bs, seq)

    def _interest(self, doc_emb, doc_mask, hist):
        domains = [self.review_agg(doc_emb, doc_mask),
                   self.history_agg(self.item_embs[hist.long()], tc.where(hist >= 1, 1.0, 0.0))]
        user_emb = self.interest_proj(*domains)
        interest = self.interest_norm(tc.mm(user_emb, self.asp_embs)).unsqueeze(1)
        return tc.sigmoid(interest) * self.asp_free # (bs, K)

    def _mention(self, iemb):
        bs, ts, dim = iemb.shape
        mention = self.mention_proj(iemb.reshape(-1, dim))
        if self.training and "mention_proj" not in self._cache:
            self._cache["mention_proj"] = mention.reshape(bs, ts, -1)
        mention = tc.nn.functional.normalize(mention, dim=-1)
        aspects = tc.nn.functional.normalize(self.asp_embs, dim=0)
        mention = tc.mm(mention, aspects).reshape(bs, ts, -1)
        return tc.softmax(mention, -1) # (bs, ts, K)
    
    def compute_loss(self, ytrue, ypred): 
        ypred = ypred.to(self.device).float()
        ytrue = ytrue.to(self.device).float()
        loss = self.lossfn(ypred, ytrue)
        if self.training:
            loss = loss +\
                   self.gamma1 * self._contrastive() +\
                   self.gamma2 * self._rate_reduction() +\
                   self.gamma3 * self._orthogonal()
        return loss 

    def _orthogonal(self):
        tmp = tc.nn.functional.normalize(self.asp_embs, dim=0)
        tmp = tc.mm(tmp.T, tmp) - self.asp_eye
        return (tmp ** 2).sum() / (tmp.shape[1] ** 2 - tmp.shape[1])

    def _rate_reduction(self):
        X, S, M = self._cache["mention_proj"], self._cache["item_ids"], self._cache["item_mask"]
        X = X * M.unsqueeze(-1)
        bs, ts, dim = X.shape
        with tc.no_grad():
            E = self.bert.encoder.embeddings.word_embeddings.weight[S.long()]
            E = tc.nn.functional.normalize(E, dim=-1) * M.unsqueeze(-1)
            A = tc.where(tc.bmm(E, E.transpose(2, 1)) >= self.threshold, 1.0, 0.0).float()           
        
            # N=(bs,): numbers of words for each sample, adding 1 to prevent overfloatting
            ones = tc.ones((bs,), device=self.device)
            N = M.sum(axis=-1) - 1.0       
            N = tc.where(N > 0, N, ones)
            I = tc.eye(dim, device=self.device)
            scaler = 2 * tc.log(ones * 10.0)
            infinity = ones - 1e5
            smallest = ones * 1e-5
        
        # calculate coding rate for the whole document over samples
        covar = self.beta * dim * tc.bmm(X.transpose(2, 1), X) / (N.reshape(-1, 1, 1) * self.eps)
        entire_rate = 0.5 * tc.slogdet(I + covar).logabsdet / self.beta

        # calculate coding rate for the group words over samples        
        maxN = int(max(N.cpu().tolist())) - 1
        samples = random.sample(range(1, maxN), int(self.sampling * maxN)) # we only calculate the scores for parts of words
        groups_rate = ones - 1.0
        for gid in samples:
            word_adj = A[:, gid]
            adj = tc.diag_embed(word_adj) # (bs, ts, ts)
            trace = word_adj.sum(axis=-1) + 1.0 / N # (bs,): add 1 / N to prevent overfloating
            covar = dim * tc.bmm(tc.bmm(X.transpose(2, 1), adj), X) / (trace.reshape(-1, 1, 1) * self.eps)
            score = trace * tc.slogdet(I + covar).logabsdet / (N * 2.0)
            score = tc.where(tc.isinf(score), infinity, score)
            score = tc.where(tc.isnan(score), smallest, score)
            groups_rate = groups_rate + score / N
        return - (entire_rate - groups_rate).mean()

    def _contrastive(self):
        ids = self._cache["current_ids"].to(self.device)
        mask = self._cache["current_mask"].to(self.device)
        hist = self._cache["user_history"]
        emb = self._encode(ids, mask)
        sent = self._sentiment_analysis(emb)
        gate = (self._mention(emb) * self._interest(emb, mask, hist)).sum(axis=-1) * mask
        score = average(sent * gate, gate).squeeze().detach()
        return ((self._cache["major_score"] - score) ** 2).mean()
    
