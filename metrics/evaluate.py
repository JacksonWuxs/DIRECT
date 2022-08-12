import random

import tqdm
import numpy as np
import torch as tc
from collections import defaultdict

from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             f1_score,
                             ndcg_score,
                             mean_squared_error,
                             mean_absolute_error)



class CTREvaluator:
    def __init__(self, threshold):
        self.threshold = threshold
        self.hist_real = []
        self.hist_pred = []

    def evaluate(self, model, data):
        model.eval()
        with tc.no_grad():
            loss = []
            debug = [[] for _ in range(5)]
            for batch in data:
                pred = model(**batch).reshape(-1,)
                self.update(batch["score"], pred)
                for label, score in zip(batch["score"].cpu().numpy().tolist(),
                                        pred.detach().cpu().numpy().tolist()):
                    debug[int(label) - 1].append(score)
                loss.append(model.compute_loss(batch["score"], pred).cpu())
        print([np.mean(_) for _ in debug])
        auc, acc, f1, mse, mae = self.score()
        model.train()
        return auc, acc, f1, mse, mae, np.mean(loss)

    def update(self, real_score, pred_score):
        self.hist_pred.extend(self._check_is_python(pred_score))
        self.hist_real.extend(self._check_is_python(real_score))

    def _check_is_python(self, data):
        if isinstance(data, (tc.Tensor,)):
            data = data.cpu().numpy().flatten().tolist()
        assert all(isinstance(_, (float, int)) for _ in data)
        return data

    def score(self):
        real_score, pred_score = np.array(self.hist_real), np.array(self.hist_pred)
        pred_label = np.where(pred_score >= self.threshold, 1.0, 0.0)
        real_label = np.where(real_score >= self.threshold, 1.0, 0.0)
        self.hist_pred.clear()
        self.hist_real.clear()
        return (roc_auc_score(real_label, pred_score),
                accuracy_score(real_label, pred_label),
                f1_score(real_label, pred_label),
                mean_squared_error(real_score, pred_score),
                mean_absolute_error(real_score, pred_score))


def combine_records(data):
    records = defaultdict(set)
    for uid, iid, _ in data:
        records[uid].add(iid)
    return records



class TestDataset(tc.utils.data.Dataset):
    def __init__(self, uid, exclude_iids, meta_index):
        self.uid = uid
        self.meta = meta_index
        self.iid = [_ for _ in range(len(meta_index.items)) if _ not in exclude_iids]

    def __len__(self):
        return len(self.iid)

    def __getitem__(self, idx):
        return self.meta.get_feed_dict(self.uid, self.iid[idx])


class TopKEvaluator:
    def __init__(self, meta, train, valid, k, threshold=4.0, batch_size=32):
        self.threshold = threshold
        self.batchsize = batch_size 
        self.top_k = k
        train_record = combine_records(train)
        self.valid_record = combine_records(valid)
        self.test_uids = list(set(train_record.keys()) & set(self.valid_record.keys()))
        random.shuffle(self.test_uids)
        self.test_data = [TestDataset(_, train_record[_], meta) for _ in self.test_uids[:100]]

    def evaluate(self, model):
        precision = {k: [] for k in self.top_k}
        recall = {k: [] for k in self.top_k}
        ndcg = {k: [] for k in self.top_k}
        model.eval()
        with tc.no_grad():
            for dataset in tqdm.tqdm(self.test_data):
                item_scores = []
                data = tc.utils.data.DataLoader(dataset, batch_size=self.batchsize, shuffle=False, num_workers=6, pin_memory=True)
                for batch in tqdm.tqdm(data, total=len(dataset) // self.batchsize):
                    pred_label = tc.where(model(**batch) >= self.threshold, 1.0, 0.0)
                    item_scores.extend(pred_label.detach().cpu().numpy().flatten().tolist())

                pair_sorted = sorted(enumerate(item_scores), key=lambda x: x[1], reverse=True)
                item_sorted = [_[0] for _ in pair_sorted]
                score_sorted = [_[1] for _ in pair_sorted]
                user_clicked = self.valid_record[dataset.uid]
                
                for k in self.top_k:
                    num_hit = len(set(item_sorted[:k]) & user_clicked)
                    precision[k].append(num_hit / k)
                    recall[k].append(num_hit / len(user_clicked))
                    ndcg[k].append(ndcg_score([[1.0 if _ in user_clicked else 0.0 for _ in item_sorted]],
                                              [score_sorted], k=k))
        model.train()
        return {"top_k": self.top_k,
                "precision": [np.mean(precision[k]) for k in self.top_k],
                "recall": [np.mean(recall[k]) for k in self.top_k],
                "ndcg": [np.mean(ndcg[k]) for k in self.top_k],}           
