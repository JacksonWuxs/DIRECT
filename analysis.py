import reset_seed

SEED = 37869
reset_seed.frozen(SEED)

import os
import math
import copy

import pytorch_warmup
import numpy as np
import torch as tc
import sklearn
import tqdm


from analysts.aspects import do_aspect_analysis
from analysts.case_study import do_case_study
from metrics.evaluate import TopKEvaluator, CTREvaluator
from models.DIRECT import  DIRECT
from datas.dataset import Dataset, MetaIndex
from datas.logger import print
from datas.preprocess import initialize_dataset


plm = "prajjwal1/bert-small"
amazon = ("reviewerID", "asin", "reviewText", "overall")
yelp = ("user_id", "business_id", "text", "stars")

datafiles = [
             "./datasets/reviews_Toys_and_Games_5.json",
             "./datasets/reviews_Clothing_Shoes_and_Jewelry_5.json",
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
             "len_sent": 128,
             "num_hist": 20,
             "len_doc": 510,
             "drop_stopword": False,
             "cache_freq": 1,
             "keep_head": 1.0,
             },
    "data": {"sampling": None,
             "cache_freq": 1}}



MODEL_CONFIG = {
    "plm": plm,
    "dropout": 0.3,
    "aspc_num": 5, 
    "aspc_dim": 64,
    "gamma1": 0.0,
    "gamma2": 0.0,
    "gamma3": 0.0,
    "beta": 0.1,
    "sampling": 0.1,
    "threshold": 0.2,
    "user_emb": "gate",
    "device": "cuda",
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
    return train_meta, subsets




if __name__ == "__main__":
    for datafile in datafiles:
        format_ = yelp if "yelp" in datafile else amazon
        meta, datas = get_subsets(datafile, format_, DATA_CONFIG)
        model = DIRECT(user_num=len(meta.users), item_num=len(meta.items), **MODEL_CONFIG)
        best_model = "outputs/seed%d_DIRECT_%s.pth" % (SEED, os.path.split(datafile)[-1])
        model.load_state_dict(tc.load(best_model))
        print("Model has been reloaded: %s" % best_model)
        do_case_study(meta, datas[0], model, "cuda", num_user=30, num_item=30, topK=0.1)
        do_aspect_analysis(meta, datas[0], model, "cuda", datafile.split(r"/")[-1].replace(".json", ''))
