import json
import os
import re
import time
import pickle
import collections

import tqdm
import torch as tc
import numpy as np

from .tokenizer import Tokenizer
from .lookup import LookupTable


class CorpusSearchIndex:
    def __init__(self, file_path, cache_freq=5, sampling=None):
        assert os.path.exists(file_path)
        self.datafile = file_path
        self.cache_freq = cache_freq
        self.lookup, self._numrow = [0], 0
        with open(file_path, encoding="utf8") as f:
            while self._numrow != sampling:
                row = f.readline()
                if len(row) == 0:
                    break
                self._numrow += 1
                if self._numrow % cache_freq == 0:
                    self.lookup.append(f.tell())

    def __iter__(self):
        with open(self.datafile, encoding="utf8") as f:
            for row in f:
                yield row.strip()

    def __len__(self):
        return self._numrow

    def __getitem__(self, index):
        cacheid = index // self.cache_freq
        with open(self.datafile, encoding="utf8") as f:
            f.seek(self.lookup[cacheid])
            for idx, row in enumerate(f, cacheid * self.cache_freq):
                if idx == index:
                    return row.strip()
        raise IndexError("Index %d is out of boundary" % index)


class DocumentDataset(tc.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_len, keep_head, cache_freq):
        super(tc.utils.data.Dataset).__init__()
        self.tokenizer = Tokenizer(tokenizer, max_len, keep_head)
        self.documents = CorpusSearchIndex(file_path, cache_freq)

    def __getitem__(self, idx):
        doc = self.documents[idx].replace("\t", " ").strip()
        doc_id, doc_mask = self.tokenizer.transform(doc.split())
        return np.array(doc_id), np.array(doc_mask)

    def __len__(self):
        return len(self.documents)


class MetaIndex:
    def __init__(self, root, tokenizer, num_sent, len_sent,
                 num_hist, len_doc, users=None, items=None, drop_stopword=False, keep_head=0.7, cache_freq=5):
        self.num_sent = num_sent
        self.num_hist = num_hist
        self.len_doc = len_doc
        self.root = root
        self.users = users if users else LookupTable.from_txt(root + r"/user_idx.txt")
        self.items = items if items else LookupTable.from_txt(root + r"/item_idx.txt")
        self.doc_tokenizer = Tokenizer(tokenizer, len_doc, keep_head, drop_stopword) if len_doc else None
        self.sent_tokenizer = Tokenizer(tokenizer, len_sent, keep_head, drop_stopword) if num_sent else None
        self.user_doc = CorpusSearchIndex(root + r"/user_doc.txt", cache_freq)
        self.item_doc = CorpusSearchIndex(root + "/item_doc.txt", cache_freq)
        self.user_stars = CorpusSearchIndex(root + "/user_score.txt", cache_freq)
        self.user_hist = CorpusSearchIndex(root + "/user_history.txt", cache_freq)
        self.item_hist = CorpusSearchIndex(root + "/item_history.txt", cache_freq)

    def get_feed_dict(self, uid, iid, current_review=""):
        if isinstance(uid, str):
            uid = self.users[uid]
        if isinstance(iid, str):
            iid = self.items[iid]
        rating_hist = self._get_history(self.user_stars[uid], float, False)
        user_hist = self._get_history(self.user_hist[uid], self.items.__getitem__, True)
        item_hist = self._get_history(self.item_hist[iid], self.users.__getitem__, True)
        
        user_review = self.user_doc[uid].replace(current_review, "")
        item_review = self.item_doc[iid].replace(current_review, "")
        current_id, current_mask = self._get_doc_reviews(current_review)
        user_sent_id, user_sent_mask = self._get_sent_reviews(user_review)
        user_doc_id, user_doc_mask = self._get_doc_reviews(user_review)
        item_sent_id, item_sent_mask = self._get_sent_reviews(item_review)
        item_doc_id, item_doc_mask = self._get_doc_reviews(item_review)
        return {"uid": uid,
                "iid": iid,
                "current_ids": current_id,
                "current_mask": current_mask,
                
                "user_hist_ids": user_hist,
                "item_hist_ids": item_hist,
                
                "user_hist_rate": rating_hist,
                
                "user_doc_ids": user_doc_id,
                "user_doc_mask": user_doc_mask,
                "user_sent_ids": user_sent_id,
                "user_sent_mask": user_sent_mask,
                "item_doc_ids": item_doc_id,
                "item_doc_mask": item_doc_mask,
                "item_sent_ids": item_sent_id,
                "item_sent_mask": item_sent_mask,
                }


    def _get_history(self, data, func, add_bias=True, padding=0):
        hist = [func(_) for _ in data.split("\t")[:self.num_hist] if len(_) > 0]
        if add_bias:
            hist = [_ + 1 for _ in hist]
        return np.array(hist + [padding] * (self.num_hist - len(hist)))
        
    def _get_sent_reviews(self, reviews):
        if self.num_sent is None:
            return -1, -1
        sent_id, sent_mask = [], []
        for review in reviews.split("\t"):
            if len(review) == 0:
                continue
            ids, mask = self.sent_tokenizer.transform(review.split())
            sent_id.append(ids)
            sent_mask.append(mask)
            if len(sent_id) == self.num_sent:
                break
        sent_id.extend([[0] * (2 + self.sent_tokenizer.max_word)] * (self.num_sent - len(sent_id)))
        sent_mask.extend([[0] * (2 + self.sent_tokenizer.max_word)] * (self.num_sent - len(sent_mask)))
        return np.vstack(sent_id), np.vstack(sent_mask)

    def _get_doc_reviews(self, reviews):
        if self.len_doc is None:
            return -1, -1
        doc_id, doc_mask = self.doc_tokenizer.transform(reviews.replace("\t", " ").split())
        return np.array(doc_id), np.array(doc_mask)
    

class Dataset(tc.utils.data.Dataset):
    def __init__(self, root, subset, format_, metaset, cache_freq, sampling):
        super(tc.utils.data.Dataset).__init__()
        self.keys, self.meta = format_, metaset
        self.users, self.items = self.meta.users, self.meta.items
        self.data = CorpusSearchIndex(root + r"/%s.json" % subset, cache_freq, sampling)

    def __iter__(self):
        for row in self.data:
            user, item, review, score = self.get_record(row)
            recommend = 1. if score >= 4.0 else 0.
            yield self.users[user], self.items[item], review, recommend

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, review, score = self.get_record(self.data[idx])
        recommend = 1. if score >= 4.0 else 0.
        feed = self.meta.get_feed_dict(user, item, review)
        feed.update({"score": score,
                     "recommend": recommend})
        return feed

    @property
    def num_user(self):
        return len(self.users)

    @property
    def num_item(self):
        return len(self.items)

    def get_record(self, row):
        row = json.loads(row)
        return (row[_] for _ in self.keys)

