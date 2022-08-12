import random
import heapq
import collections

import torch as tc
import numpy as np



CaseStudy = collections.namedtuple("CaseStudy", ["uid", "iid", "interest",
                                                 "HistSent", "HistGate", "HistDoc", "HistReview",
                                                 "TgtSent", "TgtGate", "TgtDoc",
                                                 "golden", "predict",
                                                 "preference", "bias"])


def average(emb, mask=None):
    assert len(mask.shape) == 2
    if len(emb.shape) == 2:
        emb = emb.unsqueeze(-1)
    assert emb.shape[0] == mask.shape[0] and emb.shape[1] == mask.shape[1]
    return (mask.unsqueeze(-1) * emb).sum(axis=1) / (5e-7 + mask.sum(axis=-1, keepdims=True))





class DecisionMakingAnalyzer:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(device)

    def analysis(self, meta, data, num_user=1, num_item=2, topK=30):
        uids = random.sample(range(len(meta.users)), num_user)
        iids = []
        for uid in uids:
            history = [meta.items[_] for _ in meta.user_hist[uid].strip().split("\t")]
            iids.append(random.sample(history, min(len(history), num_item)))
        interests = dict((uid, self._analysis_user_interest(uid, meta)) for uid in uids)
        samples = self._findout_reviews(meta, data, uids, iids)
        results = []
        for uid, iid, source, score in samples:
            ids, mask, review = self._prepare_text(source, meta)
            sentiment_tgt = self._analysis_doc_sentiment(ids, mask)
            mention_tgt = self._analysis_doc_mention(ids, mask)
            activate_tgt = self._activate_topk(mention_tgt, interests[uid], review, topK)

            ids, mask, document = self._prepare_text(meta.item_doc[iid].replace(source, ""), meta)
            sentiment_hist = self._analysis_doc_sentiment(ids, mask)
            mention_hist = self._analysis_doc_mention(ids, mask)
            activate_hist = self._activate_topk(mention_hist, interests[uid], document, topK)
            gate = (mention_hist * interests[uid]).sum(axis=-1) * mask
            major_score = average(sentiment_hist * gate, gate).squeeze()
            bias_score = self.model.bias(tc.tensor([uid]).to(self.device),
                                    tc.tensor([iid]).to(self.device))
            results.append(CaseStudy(meta.users[uid], meta.items[iid], interests[uid].squeeze().cpu().tolist(),
                                     sentiment_hist.squeeze().cpu().tolist(), activate_hist, document, meta.item_doc[iid].replace(source, ""),
                                     sentiment_tgt.squeeze().cpu().tolist(), activate_tgt, review,
                                     score, (bias_score + major_score).squeeze().cpu().tolist(),
                                     major_score.squeeze().cpu().tolist(), bias_score.squeeze().cpu().tolist()
                                     ))
        return results           

    def render(self, result):
        assert isinstance(result, CaseStudy)
        text = "UID=%s | IID=%s | Gold=%.0f | Pred=%.2f | Pref=%.2f | Bias=%.2f" % (
             result.uid, result.iid, result.golden, result.predict, result.preference, result.bias)
        size = len(text)
        text = "=" * size + "\n" + text + "\n" + "-" * size + "\n"
        text += "Interest: " + "|".join("%d: %.4f" % (i, p) for i, p in enumerate(result.interest[1:], 1)) + "\n"
        text += "Target: " + " ".join(_[0] if _[1] == 0.0 else "%s[%.2f|%.2f]" % _ for _ in zip(result.TgtDoc,
                                                                                                result.TgtGate,
                                                                                                result.TgtSent)) + "\n"
        text += "Document: " + " ".join(_[0] if _[1] == 0.0 else "%s[%.2f|%.2f]" % _ for _ in zip(result.HistDoc,
                                                                                                  result.HistGate,
                                                                                                  result.HistSent)) + "\n"
        text += "Reviews: " + result.HistReview.replace("\t", "\n")
        text += "=" * size
        return text.replace(" [PAD]", "")
        
        
    def _findout_reviews(self, meta, data, uids, iids):
        pairs = set((uid, iid) for uid, cases in zip(uids, iids) for iid in cases)
        for idx, (uid, iid, _, _) in enumerate(data):
            if (uid, iid) in pairs:
                _, _, review, score = data.get_record(data.data[idx])
                yield uid, iid, review, score
        

    def _analysis_user_interest(self, uid, meta):
        ids, mask, _ = self._prepare_text(meta.user_doc[uid], meta)
        hist = meta._get_history(meta.user_hist[uid], meta.items.__getitem__, True)
        embs = self.model._encode(ids, mask)
        return self.model._interest(embs, mask, tc.tensor(hist).unsqueeze(0).to(self.device))

    def _analysis_doc_sentiment(self, ids, mask):
        embs = self.model._encode(ids, mask)
        return self.model._sentiment_analysis(embs) * mask

    def _analysis_doc_mention(self, ids, mask):
        return self.model._mention(self.model._encode(ids, mask)) * mask.unsqueeze(-1)

    def _prepare_text(self, document, meta):
        ids, mask, _ = meta._get_doc_reviews(document)
        words = self.model.bert.tokenizer.convert_ids_to_tokens(ids.tolist())
        ids = tc.tensor(ids).unsqueeze(0).to(self.device)
        mask = tc.tensor(mask).unsqueeze(0).to(self.device)
        return ids, mask, words

    def _activate_topk(self, mention, interest, texts, topk):
        activate = (mention * interest).sum(axis=-1).squeeze().cpu().tolist() # (ts,)
        if topk < 1:
            topk = int(topk * len(activate))
        topk_val = heapq.nlargest(topk, activate)[-1]
        max_val = max(activate)
        return [val if val >= topk_val and not word.startswith("[") else 0.0 for (val, word) in zip(activate, texts)]


def do_case_study(meta, data, model, device, num_user=30, num_item=10, topK=0.3):
    analyst = DecisionMakingAnalyzer(model, device)
    results = analyst.analysis(meta, data, num_user, num_item, topK)
    for result in results:
        print("\n")
        print(analyst.render(result))
    return results
