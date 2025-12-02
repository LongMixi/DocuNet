import os
import json
import numpy as np

rel2id = json.load(open('./meta/rel2id.json', 'r'))
id2rel = {v: k for k, v in rel2id.items()}


def to_official(preds, features):
    """
    Convert model predictions into DocRED official format.
    preds: numpy array (N, num_relations)
    features: list of dicts, each with hts and title
    """
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"]] * len(hts)

    assert preds.shape[0] == len(h_idx), \
        f"Prediction length {preds.shape[0]} != feature pairs {len(h_idx)}"

    res = []
    for i in range(preds.shape[0]):
        pred_vec = preds[i]
        pred_labels = np.nonzero(pred_vec)[0].tolist()

        for p in pred_labels:
            if p != 0:  # relation 0 = NA
                res.append({
                    "title": title[i],
                    "h_idx": h_idx[i],
                    "t_idx": t_idx[i],
                    "r": id2rel[p],
                })

    return res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        triples = json.load(open(fact_file_name))
        return set(tuple(x) for x in triples)

    fact_in_train = set()
    ori_data = json.load(open(data_file_name))

    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))
    return fact_in_train


def official_evaluate(tmp, path):
    if len(tmp) == 0:
        return 0, 0, 0, 0, 0, 0

    truth_dir = os.path.join(path, 'ref')
    os.makedirs(truth_dir, exist_ok=True)

    fact_in_train_annot = gen_train_facts(os.path.join(path, "train_annotated.json"), truth_dir)
    fact_in_train_dist = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, "dev.json")))

    std = {}
    tot_evidences = 0
    title2vec = {}

    for x in truth:
        title = x['title']
        title2vec[title] = x['vertexSet']

        for label in x['labels']:
            r, h, t = label['r'], label['h'], label['t']
            std[(title, r, h, t)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    # Sort predictions
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], str(x['r'])))

    # Remove duplicates
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x, y = tmp[i], tmp[i-1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != \
           (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(x)

    # Begin evaluation
    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annot = 0
    correct_in_train_dist = 0

    for x in submission_answer:
        title, h_idx, t_idx, r = x['title'], x['h_idx'], x['t_idx'], x['r']
        if title not in title2vec:
            continue

        vertexSet = title2vec[title]
        evi = set(x.get("evidence", []))
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            gt_evi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(gt_evi & evi)

            # check train facts
            in_annot = False
            in_dist = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annot:
                        in_annot = True
                    if (n1['name'], n2['name'], r) in fact_in_train_dist:
                        in_dist = True

            if in_annot:
                correct_in_train_annot += 1
            if in_dist:
                correct_in_train_dist += 1

    # Metrics
    re_p = correct_re / len(submission_answer)
    re_r = correct_re / len(std)
    re_f1 = 2 * re_p * re_r / (re_p + re_r + 1e-10)

    evi_p = correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = correct_evidence / tot_evidences
    evi_f1 = 2 * evi_p * evi_r / (evi_p + evi_r + 1e-10)

    # ignore train
    re_p_annot = (correct_re - correct_in_train_annot) / \
                 (len(submission_answer) - correct_in_train_annot + 1e-5)
    re_f1_annot = 2 * re_p_annot * re_r / (re_p_annot + re_r + 1e-10)

    re_p_dist = (correct_re - correct_in_train_dist) / \
                (len(submission_answer) - correct_in_train_dist + 1e-5)
    re_f1_dist = 2 * re_p_dist * re_r / (re_p_dist + re_r + 1e-10)

    return re_f1, evi_f1, re_f1_annot, re_f1_dist, re_p, re_r
