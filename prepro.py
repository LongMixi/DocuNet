import os
import pickle
import json
from tqdm import tqdm
from torch.utils import data
import random
import numpy as np

docred_rel2id = json.load(open('./meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    """Split list l into consecutive chunks of size n"""
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res.append(l[i:i + n])
    return res


class ReadDataset:
    def __init__(self, dataset: str, tokenizer, max_seq_length: int = 1024, transformers: str = 'bert'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transformers = transformers

    def read(self, file_in: str):
        save_file = f"{os.path.splitext(file_in)[0]}_{self.transformers}_{self.dataset}.pkl"
        if self.dataset == 'docred':
            return read_docred(self.transformers, file_in, save_file, self.tokenizer, self.max_seq_length)
        elif self.dataset == 'cdr':
            return read_cdr(file_in, save_file, self.tokenizer, self.max_seq_length)
        elif self.dataset == 'gda':
            return read_gda(file_in, save_file, self.tokenizer, self.max_seq_length)
        else:
            raise RuntimeError(f"No read function for dataset: {self.dataset}")


# ------------------- DocRED -------------------
def read_docred(transformers, file_in, save_file, tokenizer, max_seq_length=1024):
    if not file_in or not os.path.exists(file_in):
        raise FileNotFoundError(f"{file_in} not found.")

    if os.path.exists(save_file):
        with open(save_file, 'rb') as fr:
            features = pickle.load(fr)
        print(f"Loaded preprocessed data from {save_file}.")
        return features

    with open(file_in, 'r') as fh:
        data = json.load(fh)

    features = []
    max_len = 0
    up512_num = 0
    i_line = 0
    pos_samples = 0
    neg_samples = 0

    if transformers == 'bert':
        entity_type = ["-", "ORG", "-", "LOC", "-", "TIME", "-", "PER", "-", "MISC", "-", "NUM"]
    else:
        entity_type = ["*"]

    for sample in tqdm(data, desc="Processing DocRED"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        start2type = {(m["sent_id"], m["pos"][0]): m["type"] for e in entities for m in e}
        end2type = {(m["sent_id"], m["pos"][1]-1): m["type"] for e in entities for m in e}

        # Flatten tokens and add special entity markers
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token, add_special_tokens=False)

                if (i_s, i_t) in start2type:
                    mention_type = start2type[(i_s, i_t)]
                    if transformers == 'bert':
                        special_token_i = entity_type.index(mention_type)
                        tokens_wordpiece = [f"[unused{special_token_i}]"] + tokens_wordpiece
                    else:
                        tokens_wordpiece = ["*"] + tokens_wordpiece

                if (i_s, i_t) in end2type:
                    mention_type = end2type[(i_s, i_t)]
                    if transformers == 'bert':
                        special_token_i = entity_type.index(mention_type) + 50
                        tokens_wordpiece = tokens_wordpiece + [f"[unused{special_token_i}]"]
                    else:
                        tokens_wordpiece = tokens_wordpiece + ["*"]

                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        max_len = max(max_len, len(sents))
        if len(sents) > 512:
            up512_num += 1

        # Build triples
        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                h, t = label['h'], label['t']
                r = docred_rel2id[label['r']]
                if (h, t) not in train_triple:
                    train_triple[(h, t)] = [{'relation': r, 'evidence': label['evidence']}]
                else:
                    train_triple[(h, t)].append({'relation': r, 'evidence': label['evidence']})

        # Entity positions
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end))

        # Relations & HTs
        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        # Negative samples
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        # Limit to max_seq_length
        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = {
            'input_ids': input_ids,
            'entity_pos': entity_pos,
            'labels': relations,
            'hts': hts,
            'title': sample['title'],
        }
        features.append(feature)
        i_line += 1

    print(f"# of documents: {i_line}.")
    print(f"# of positive examples: {pos_samples}.")
    print(f"# of negative examples: {neg_samples}.")
    print(f"# of examples len>512: {up512_num}, max len is {max_len}.")

    with open(save_file, 'wb') as fw:
        pickle.dump(features, fw)
    print(f"Finished reading {file_in} and save to {save_file}.")
    return features


# ------------------- CDR -------------------
def read_cdr(file_in, save_file, tokenizer, max_seq_length=1024):
    if os.path.exists(save_file):
        with open(save_file, 'rb') as fr:
            features = pickle.load(fr)
        print(f"Loaded preprocessed data from {save_file}.")
        return features

    features = []
    pmids = set()
    maxlen = 0

    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for line in tqdm(lines, desc="Processing CDR"):
            line = line.rstrip().split('\t')
            pmid = line[0]
            if pmid in pmids:
                continue
            pmids.add(pmid)
            text = line[1]
            prs = chunks(line[2:], 17)
            ent2idx = {}
            train_triples = {}
            entity_pos = set()

            for p in prs:
                for start, end, tpy in [(p[8], p[9], p[7]), (p[14], p[15], p[13])]:
                    start = list(map(int, start.split(':')))
                    end = list(map(int, end.split(':')))
                    for s, e in zip(start, end):
                        entity_pos.add((s, e, tpy))

            # Tokenize sentences
            sents = [t.split(' ') for t in text.split('|')]
            new_sents = []
            sent_map = {}
            i_t = 0
            for sent in sents:
                for token in sent:
                    tokens_wordpiece = tokenizer.tokenize(token, add_special_tokens=False)
                    for start, end, tpy in entity_pos:
                        if i_t == start:
                            tokens_wordpiece = ["*"] + tokens_wordpiece
                        if i_t + 1 == end:
                            tokens_wordpiece = tokens_wordpiece + ["*"]
                    sent_map[i_t] = len(new_sents)
                    new_sents.extend(tokens_wordpiece)
                    i_t += 1
                sent_map[i_t] = len(new_sents)
            sents = new_sents

            entity_pos_list = []
            for p in prs:
                if p[0] == "not_include":
                    continue
                if p[1] == "L2R":
                    h_id, t_id = p[5], p[11]
                    h_start, t_start = p[8], p[14]
                    h_end, t_end = p[9], p[15]
                else:
                    t_id, h_id = p[5], p[11]
                    t_start, h_start = p[8], p[14]
                    t_end, h_end = p[9], p[15]

                h_start = list(map(int, h_start.split(':')))
                h_end = list(map(int, h_end.split(':')))
                t_start = list(map(int, t_start.split(':')))
                t_end = list(map(int, t_end.split(':')))

                h_start = [sent_map[idx] for idx in h_start]
                h_end = [sent_map[idx] for idx in h_end]
                t_start = [sent_map[idx] for idx in t_start]
                t_end = [sent_map[idx] for idx in t_end]

                if h_id not in ent2idx:
                    ent2idx[h_id] = len(ent2idx)
                    entity_pos_list.append(list(zip(h_start, h_end)))
                if t_id not in ent2idx:
                    ent2idx[t_id] = len(ent2idx)
                    entity_pos_list.append(list(zip(t_start, t_end)))

                h_id, t_id = ent2idx[h_id], ent2idx[t_id]
                r = cdr_rel2id[p[0]]
                if (h_id, t_id) not in train_triples:
                    train_triples[(h_id, t_id)] = [{'relation': r}]
                else:
                    train_triples[(h_id, t_id)].append({'relation': r})

            relations, hts = [], []
            for h, t in train_triples.keys():
                relation = [0] * len(cdr_rel2id)
                for mention in train_triples[h, t]:
                    relation[mention["relation"]] = 1
                relations.append(relation)
                hts.append([h, t])

            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {
                    'input_ids': input_ids,
                    'entity_pos': entity_pos_list,
                    'labels': relations,
                    'hts': hts,
                    'title': pmid,
                }
                features.append(feature)
            maxlen = max(maxlen, len(sents))

    print(f"Number of documents: {len(features)}, max len: {maxlen}")
    with open(save_file, 'wb') as fw:
        pickle.dump(features, fw)
    return features


# ------------------- GDA -------------------
def read_gda(file_in, save_file, tokenizer, max_seq_length=1024):
    # Structure is similar to read_cdr, just using gda_rel2id
    if os.path.exists(save_file):
        with open(save_file, 'rb') as fr:
            features = pickle.load(fr)
        print(f"Loaded preprocessed data from {save_file}.")
        return features

    # Reuse read_cdr code, but replace cdr_rel2id -> gda_rel2id
    features = read_cdr(file_in, save_file, tokenizer, max_seq_length)
    for f in features:
        # adjust relations size to match gda_rel2id
        if f['labels'][0]:
            f['labels'] = [l[:len(gda_rel2id)] for l in f['labels']]
    return features
