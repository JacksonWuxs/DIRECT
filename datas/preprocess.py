import os
import random
import json
import collections
import shutil
import multiprocessing

import tqdm

from .logger import print
from .lookup import LookupTable
from .tokenizer import Tokenizer


def split_dataset(root, key, valid=0.1, test=0.2, force_split=False, seed=14):
    if force_split or not os.path.exists(root + "/train.json"):
        random.seed(seed)
        with open(root + ".json", encoding="utf8") as src:
            users = collections.defaultdict(list)
            while True:
                position = src.tell()
                row = src.readline()
                if len(row) == 0:
                    break
                info = json.loads(row)
                users[info[key]].append(position)

            with open(root + "/train.json", "w", encoding="utf8") as train_file, \
                 open(root + "/valid.json", "w", encoding="utf8") as valid_file, \
                 open(root + "/test.json", "w", encoding="utf8") as test_file:
                for records in tqdm.tqdm(users.values(), total=len(users)):
                    random.shuffle(records)
                    right = int(len(records) * (1.0 - test))
                    left = int(len(records) * (1.0 - test - valid)) 
                    for parts, file in ((records[:left], train_file),
                                  (records[left:right], valid_file),
                                  (records[right:], test_file)):

                        for position in parts:
                            src.seek(position)
                            file.write(src.readline())
    numbers = []
    for _ in ["/train.json", "/valid.json", "/test.json"]:
        with open(root + _, encoding="utf8") as f:
            number = 0
            for number, _ in enumerate(f, 1):
                pass
            numbers.append(number)
    return tuple(numbers)


def groupby_record(root, key, val, suffix, order=None, force_groupby=False):
    if force_groupby or not os.path.exists(root + suffix):
        groupby = collections.defaultdict(list)
        with open(root + ".json", encoding="utf8") as data:
            for row in data:
                row = json.loads(row)
                groupby[row[key]].append(str(row[val]).replace("\n", ""))
        if os.path.split(root)[-1] in ("valid", "test"):
            with open(os.path.split(root)[0] + "/train.json", encoding="utf8") as data:
                for row in data:
                    row = json.loads(row)
                    groupby[row[key]].append(str(row[val]).replace("\n", ""))

        if order is None:
            order = sorted(groupby.keys())
        with open(root + suffix, "w", encoding="utf8") as f:
            for key in order:
                f.write("\t".join(map(str, groupby.get(key, []))) + "\n")
            

def create_lookup(root, key, min_freq, suffix, force_create=False):
    if force_create or not os.path.exists(root + suffix):
        counter = collections.Counter()
        with open(root + ".json", encoding="utf8") as f:
            for row in f:
                counter[json.loads(row)[key]] += 1
        with open(root + suffix, "w", encoding="utf8") as f:
            for key, freq in counter.items():
                if freq >= min_freq:
                    f.write(key + "\n")
    return LookupTable.from_txt(root + suffix)


class Pretokenizer(multiprocessing.Process):
    def __init__(self, pretrain, drop_stopword, drop_uncommen, input_queue, output_queue, jsonkey):
        multiprocessing.Process.__init__(self)
        self.tokenizer = Tokenizer(pretrain, drop_stopword=drop_stopword, drop_uncommen=drop_uncommen)
        self.inq = input_queue
        self.outq = output_queue
        self.key = jsonkey
        self.start()

    def run(self):
        while True:
            if not self.inq.empty():
                row = self.inq.get(True)
                if row is None:
                    self.inq.put(None)
                    break
                row = json.loads(row)
                row[self.key] = " ".join(self.tokenizer.preprocess(str(row[self.key]).replace("\n", "")))
                if "pedals" in row[self.key]:
                  raise
                self.outq.put(json.dumps(row))


class Writer(multiprocessing.Process):
    def __init__(self, root, queue):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.file = root + ".tokenized"
        self.start()

    def run(self):
        with open(self.file, "w", encoding="utf8") as f:
            while True:
                if not self.queue.empty():
                    value = self.queue.get(True)
                    if value is None:
                        break
                    f.write(value + "\n")
                

def pretokenize(root, key, num_worker, pretrain, drop_stopword, drop_uncommen, force_tokenize=False):
    if force_tokenize or not os.path.exists(root + ".tokenized"):
        with open(root, encoding="utf8") as f:
            total = 0
            for total, row in enumerate(f, 1):
                pass
            if total == 0:
                return
            inputs = multiprocessing.Queue(3 * num_worker)
            outputs = multiprocessing.Queue(5 * num_worker)
            writer = Writer(root, outputs)
            tokenizers = [Pretokenizer(pretrain, drop_stopword, drop_uncommen, inputs, outputs, key) \
                          for _ in range(num_worker)]
            #print("Begin to pre-tokenize reviews: Reviews=%d | Workers=%d" % (total, len(tokenizers)))
            f.seek(0)
            for row in tqdm.tqdm(f, total=total):
                inputs.put(row)
            inputs.put(None)
            [task.join() for task in tokenizers]
            outputs.put(None)
            writer.join(None)
        with open(root + ".tokenized", encoding="utf8") as f:
            for check, row in enumerate(f, 1):
                pass
        assert check == total
        shutil.copyfile(root + ".tokenized", root)
        #print("Finished pre-tokenized reviews")


def initialize_dataset(src, format_, pretrain, num_worker=4, valid=0.1, test=0.2, users=None, items=None,
                       min_freq=1, drop_stopword=True, drop_uncommen=True, force_init=False, dotokenize=False,
                       seed=14):
    assert src.endswith(".json")
    root = src.rsplit(".", 1)[0] + "_seed%d" % seed
    os.makedirs(root, exist_ok=True)
    shutil.copyfile(src, root + ".json")
    
    uid, iid, text, score = format_
    print("Begin to initialize dataset %s with random seed %d" % (os.path.split(root)[-1].rsplit("_", 1)[0].upper(), seed))
    if users is None:
        users = create_lookup(root, uid, min_freq, "/user_idx.txt", force_init)
        print("Created user lookup table: %d users" % len(users))
    if items is None:
        items = create_lookup(root, iid, min_freq, "/item_idx.txt", force_init)
        print("Created item lookup table: %d items" % len(items))
    numbers = split_dataset(root, uid, valid, test, force_init, seed)
    print("Splitted dataset: train=%d | valid=%d | test=%d" % numbers)
    if dotokenize:
        pretokenize(root + "/train.json", text, num_worker, pretrain, drop_stopword, drop_uncommen, force_init)
        pretokenize(root + "/valid.json", text, num_worker, pretrain, drop_stopword, drop_uncommen, force_init)
        pretokenize(root + "/test.json", text, num_worker, pretrain, drop_stopword, drop_uncommen, force_init)
    groupby_record(root, uid, text, "/user_doc.txt", users, force_init) # user document
    groupby_record(root, iid, text, "/item_doc.txt", items, force_init) # item document
    groupby_record(root, uid, score, "/user_score.txt", users, force_init) # user scores
    groupby_record(root, uid, iid, "/user_history.txt", users, force_init) # user purchase history
    groupby_record(root, iid, uid, "/item_history.txt", items, force_init) # item selling history
    return {"root": root,
            "num_user": len(users),
            "num_item": len(items),
            }
