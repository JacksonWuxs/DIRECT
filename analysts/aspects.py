import collections
import string

import nltk
import numpy as np
import torch as tc
import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def merge(d, stopwords):
    for word in list(d):
        if word.endswith("ies") and word.replace("ies", "y") in d:
            d[word.replace("ies", "y")] += d[word]
            del d[word]
        elif word.endswith("s") and word[:-1] in d:
            d[word[:-1]] += d[word]
            del d[word]
        elif word.endswith("ed") and word[:-2] in d:
            d[word[:-2]] += d[word]
            del d[word]
        elif word.endswith("ing") and word[:-3] in d:
            d[word[:-3]] += d[word]
            del d[word]
        elif word.isdigit() or\
               (word.startswith("[") and word.endswith("]")) or\
               word.startswith("#") or\
               len(word) <= 2 or\
               word.lower() in stopwords:
            del d[word]
            

class AspectsAnalyzer:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.bert = model.bert
        self.model.eval()
        self.stopwords = set(nltk.corpus.stopwords.words("english")) | set(string.punctuation + string.digits)
        self.asp = tc.nn.functional.normalize(self.model.asp_embs, dim=0)

    def analysis(self, data, batch_size=32):
        progress = tqdm.tqdm(total=len(data) // batch_size)
        data = tc.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        user_memory = [collections.Counter() for _ in range(self.model.asp_embs.shape[1])]
        item_memory = [collections.Counter() for _ in range(self.model.asp_embs.shape[1])]
        
        with tc.no_grad():
            for idx, batch in enumerate(data):
                progress.update(1)
                self._batch(user_mem=user_memory,
                            item_mem=item_memory,
                            **batch)
        for memory in item_memory:
            merge(memory, self.stopwords)
        vocab = set()
        for _ in item_memory:
            vocab.update(_)

        for word in vocab:
            freqs = sum([_[word] for _ in item_memory])
            for memory in item_memory:
                memory[word] = memory[word] / freqs
        return user_memory, item_memory

    def visualize(self, data, name, batchsize=1024, minfreq=50):
        word_embs = {}
        cls, sep = self.bert.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        with tc.no_grad():
            bar = tqdm.tqdm(total=len(data))
            for k, (_, _, review, _) in enumerate(data, 1):
                bar.update(1)
                if k == 10000:
                    break
                tokens = review.split()[:510]
                ids = tc.tensor(self.bert.tokenizer.convert_tokens_to_ids([cls] + tokens + [sep])).to(self.device)
                mask = tc.ones((len(ids),)).to(self.device)
                embs = self._encode(ids.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
                proj = self.model.mention_proj(embs).detach()
                for word, emb in zip(tokens, proj.cpu().tolist()):
                    if word not in word_embs:
                        word_embs[word] = []
                    word_embs[word].append(emb)
                    
        merge(word_embs, self.stopwords)
        aspects = self.asp.detach().T.cpu().numpy() # (K, dim)
        colors = ["black", "#2878B5", "#BEB8DC", "#FA7F6F", "#C82423", "#8ECFC9"]
        groups = [0] * len(aspects)
        embeds, words, labels = [StandardScaler().fit_transform(aspects)], [], []
        for word, emb in sorted(word_embs.items(), key=lambda pair: pair[1], reverse=True):
            emb = np.mean(emb, axis=0)
            emb = emb / (emb ** 2).sum()
            lbl = np.argmax(emb @ aspects.T)
            if groups[lbl] <= 50:
                groups[lbl] += 1
                embeds.append(emb)
                labels.append(colors[lbl])
                words.append(word)
        print(collections.Counter(labels))
        print("Selected Words:", len(embeds))
        src = StandardScaler().fit_transform(np.vstack(embeds))

        D = 10
        embeds = TSNE(perplexity=D, n_components=2, n_iter=3000, learning_rate="auto", random_state=14, n_jobs=4).fit_transform(src)
        x, y = embeds[:, 0].tolist(), embeds[:, 1].tolist()
        for i in range(len(aspects)):
            plt.scatter(x[i], y[i], s=100, alpha=1.0, c=colors[i])
            plt.annotate("ASPECT-%d" % i,
                         xy=(x[i], y[i]),
                         xytext=(3, 3),
                         fontsize=16,
                         textcoords="offset points",
                         ha="right",
                         va="bottom")
        for x, y, word, lbl in zip(x[i+1:], y[i+1:], words, labels):
            plt.scatter(x, y, s=100, alpha=1.0, c=lbl)
        figure = plt.gcf() # get current figure
        figure.set_size_inches(16, 16)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("./outputs/VisualizeAspect_%s_%s.svg" % (name, D), dpi=600, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
    def _batch(self, user_mem, item_mem, current_ids, current_mask, **args):
        text = current_ids.to(self.model.device)
        mask = current_mask.to(self.model.device)
        mention = self._analysis_mention(self._encode(text, mask))
        self._category(text, item_mem, mention, mask)

    def _encode(self, texts, mask):
        return self.model.bert(texts, mask) #(bs, seq, dim)

    def _analysis_mention(self, iemb):
        bs, ts, dim = iemb.shape
        mention = self.model.mention_proj(iemb.reshape(-1, dim))
        mention = tc.nn.functional.normalize(mention, dim=-1)
        return tc.mm(mention, self.asp).reshape(bs, ts, -1)

    def _category(self, ids, cache, distribute, mask):
        aspect = tc.argmax(distribute, -1)
        aspect = tc.where(aspect == 0, self.model.asp_embs.shape[1], aspect) * mask
        for sent_asp, sent_ids in zip(aspect.cpu().tolist(), ids.cpu().tolist()):
            for word_asp, word_txt in zip(sent_asp, self.bert.tokenizer.convert_ids_to_tokens(sent_ids)):
                if word_asp != 0:
                    cache[word_asp - 1][word_txt] += 1
        return cache
            
        
        
def do_aspect_analysis(meta, data, model, device, name, topK=100, min_freq=5):
    analyst = AspectsAnalyzer(model, device)
    analyst.visualize(data, name)
    uword, iword = analyst.analysis(data)
    normalize = 1.0 / len(iword)
    for idx, words in enumerate(iword):
        print("%d: " % idx + ",".join(_[0] for _ in words.most_common(topK) if _[1] > normalize)) 

