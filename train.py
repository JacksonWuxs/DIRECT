import math
import sys
import os
import copy
import pickle

import reset_seed

SEED = int(sys.argv[1])
CUDA = str(sys.argv[2])
reset_seed.frozen(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA 


import numpy as np
import torch as tc
import tqdm


from metrics.evaluate import TopKEvaluator, CTREvaluator
from models.DIRECT import  DIRECT
from datas.dataset import Dataset, MetaIndex, DocumentDataset
from datas.logger import print
from datas.preprocess import initialize_dataset


plm = "prajjwal1/bert-small"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")

continue_ckpt = False

datafiles = [
             "./datasets/reviews_Toys_and_Games_5.json",
             "./datasets/reviews_Video_Games_5.json",
             "./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json",
             "./datasets/yelp2019_5core.json",
             "./datasets/reviews_CDs_and_Vinyl_5.json",
             ]

DATA_CONFIG = {
    "init": {"valid": 0.1,
             "test": 0.2,
             "seed": SEED,
             "min_freq": 1,
             "pretrain": plm,
             "num_worker": 8,
             "force_init": False,
             },
    "meta": {"tokenizer": plm,
             "num_sent": None,
             "len_sent": None,
             "num_hist": 30,
             "len_doc": 510,
             "cache_freq": 1,
             "keep_head": 1.0,
             "drop_stopword": False
             },
    "data": {"sampling": None,
             "cache_freq": 1}}



MODEL_CONFIG = {
    "plm": plm,
    "dropout": 0.3,
    "aspc_num": 5, 
    "aspc_dim": 64,
    "gamma1": 5e-3,  # \Loss_c: Contrastive Training 
    "gamma2": 1e-6,  # \Omega_d: Diversity Assumption
    "gamma3": 2.5,   # \Omega_o: Orthogonal Assumption
    "beta": 0.1,
    "sampling": 0.1,
    "threshold": 0.2,
    "device": "cuda:%s" % CUDA,
    }


TRAIN_CONFIG = {
    "use_amp": False,
    "learn_rate": 1e-3,
    "batch_size": 32,
    "workers": 2,
    "num_epochs": 50,
    "decay_rate": 0.1,
    "decay_tol": 2,
    "early_stop": 2,
    "weight_decay": 1e-6,
    "optimizer": "AdamW",
    "max_norm": 1.0,
    "frozen_train_size": 30,
    "log_frequency": 200000
    }
    

def get_subsets(root, format_, configs, splits=("train", "valid", "test")):
    assert isinstance(configs, dict) and len(configs) == 3
    assert "init" in configs and "data" in configs and "meta" in configs
    configs = copy.deepcopy(configs)
    root_info = initialize_dataset(datafile, format_, dotokenize=True, **configs["init"])
    configs["init"]["valid"] = configs["init"]["test"] = 0.0
    meta = MetaIndex(root_info["root"], **configs["meta"])
    train_info = initialize_dataset(root_info["root"] + "/train.json", format_, users=meta.users, items=meta.items, **configs["init"])
    train_meta = MetaIndex(train_info["root"], users=meta.users, items=meta.items, **configs["meta"])
    subsets = [Dataset(train_info["root"], "train", format_, train_meta, **configs["data"])]
    for split in splits[1:]:
        splitfile = root_info["root"] + "/" + split + ".json"
        info = initialize_dataset(splitfile, format_, users=train_meta.users, items=train_meta.items, **configs["init"])
        tmp_meta = MetaIndex(train_info["root"], users=meta.users, items=meta.items, **configs["meta"])
        subsets.append(Dataset(info["root"], "train", format_, tmp_meta, **configs["data"]))
    documents = DocumentDataset(train_info["root"] + "/item_doc.txt",
                                configs["meta"]["tokenizer"], configs["meta"]["len_doc"],
                                configs["meta"]["keep_head"], 1)
    return train_meta, subsets, documents


def fit(datas, model, optimizer, learn_rate, batch_size, num_epochs, max_norm, log_frequency, frozen_train_size, decay_rate, decay_tol, early_stop, weight_decay, use_amp, workers):
    optimizer = getattr(tc.optim, optimizer)(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)
    progress = tqdm.tqdm(total=math.ceil(len(datas[0]) / batch_size) * num_epochs)
    scaler = tc.cuda.amp.GradScaler(enabled=use_amp)
    
    train = tc.utils.data.DataLoader(datas[0], batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid = tc.utils.data.DataLoader(datas[1], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test = tc.utils.data.DataLoader(datas[2], batch_size=batch_size, shuffle=False,  num_workers=2, pin_memory=True)
    ctr_grader = CTREvaluator(threshold=4.0)

    epoch, total_tol, current_tol = 0, 0, 0
    frozen_train = []
    best_model, best_score = "outputs/seed%d_%s_%s.pth" % (SEED, model.version, os.path.split(datafile)[-1]), float("inf")
    idx = 0
    if continue_ckpt:
        if os.path.isfile(best_model + ".trainer"):
            with open(best_model + ".trainer", "rb") as f:
                info = pickle.load(f)
                assert info["best_model"] == best_model
            model.load_state_dict(tc.load(best_model))
            epoch, idx = info["epoch"], info["index"]
            best_score, total_tol = info["best_score"], info["total_tol"]
            print("Reloaded Model Success!")
        else:
            print("Ignore reloaded model: %s" % best_model)
        
    for epoch in range(epoch + 1, num_epochs + 1):
        model.train()
        for batch in train:
            if idx <= frozen_train_size:
                frozen_train.append(batch)
            if batch["recommend"].shape[0] != batch_size:
                continue
            optimizer.zero_grad()
            if use_amp:
                with  tc.cuda.amp.autocast():
                    loss = model.compute_loss(batch["score"], model(**batch))
                scaler.scale(loss).backward()
                tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                model.compute_loss(batch["score"], model(**batch)).backward()
                tc.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()
            progress.update(1)
            idx += 1
            
        auc, acc, f1, mse, mae, loss = ctr_grader.evaluate(model, valid)
        print("Valid Epoch=%d | Accuracy=%.4f | AUC=%.4f | F1=%.4f | MSE=%.4f | MAE=%.4f" % (epoch, acc, auc, f1, mse, mae))
        if mse < best_score - 5e-5:
            current_tol = 0
            best_score = mse
            tc.save(model.state_dict(), best_model)
            with open(best_model + ".trainer", "wb") as f:
                pickle.dump({"epoch": epoch, "index": idx,
                          "best_model": best_model,
                          "best_score": best_score,
                          "total_tol": total_tol},
                         f)
        else:
            current_tol += 1
            if current_tol == decay_tol:
                print("Reducing learning rate by %.4f" % decay_rate)
                scheduler.step()
                model.load_state_dict(tc.load(best_model))
                current_tol = 0
                total_tol += 1
            if total_tol == early_stop + 1:
                print("Early stop at epoch %s with MSE=%.4f" % (epoch, mse))
                break


    print("Reload model:" + best_model)
    model.load_state_dict(tc.load(best_model))
    auc, acc, f1, mse, mae, loss = ctr_grader.evaluate(model, test)
    print("Test Epoch=%d | Accuracy=%.4f | AUC=%.4f | F1=%.4f | MSE=%.4f | MAE=%.4f" % (epoch, acc, auc, f1, mse, mae))
    return mse, model


if __name__ == "__main__":
    for datafile in datafiles:
        format_ = yelp if "yelp" in datafile else amazon
        meta, datas, item_doc = get_subsets(datafile, format_, DATA_CONFIG)
        model = DIRECT(user_num=len(meta.users),
                     item_num=len(meta.items),
                     **MODEL_CONFIG)
        model.prepare_item_embedding(item_doc)
        mse, model = fit(datas, model, **TRAIN_CONFIG)
