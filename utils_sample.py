import torch
import random
import numpy as np


def set_seed(args):
    """
    Set seed for reproducibility.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    """
    Collate function for standard DocRE datasets.
    Pads input_ids and attention_mask, returns labels and hts.
    """
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    
    entity_pos = [f["entity_pos"] for f in batch]
    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    
    return input_ids, attention_mask, labels, entity_pos, hts


def collate_fn_sample(batch, negative_alpha=8, positive_alpha=1):
    """
    Collate function for datasets with positive/negative sampling.
    Mixes positive and negative samples based on given alphas.
    """
    max_len = max(len(f["input_ids"]) for f in batch)
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    
    entity_pos = [f["entity_pos"] for f in batch]

    labels, hts = [], []
    for f in batch:
        randnum = random.randint(0, 1000000)
        pos_hts = f.get('pos_hts', [])
        pos_labels = f.get('pos_labels', [])
        neg_hts = f.get('neg_hts', [])
        neg_labels = f.get('neg_labels', [])

        if negative_alpha > 0 and neg_hts:
            random.seed(randnum)
            random.shuffle(neg_hts)
            random.seed(randnum)
            random.shuffle(neg_labels)
            lower_bound = int(max(20, len(pos_hts) * negative_alpha))
            combined_hts = pos_hts * positive_alpha + neg_hts[:lower_bound]
            combined_labels = pos_labels * positive_alpha + neg_labels[:lower_bound]
            hts.append(combined_hts)
            labels.append(combined_labels)
        else:
            hts.append(pos_hts)
            labels.append(pos_labels)

    return input_ids, attention_mask, labels, entity_pos, hts
