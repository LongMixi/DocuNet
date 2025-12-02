import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model_balanceloss import DocREModel
from utils_sample import set_seed, collate_fn
from prepro import read_cdr, read_gda
import time


def train(args, model, train_features, dev_features, test_features):
    def logging(s):
        print(s)
        if args.log_dir:
            os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)
            with open(args.log_dir, 'a+') as f_log:
                f_log.write(s + '\n')

    def finetune(features, optimizer, num_epoch, num_steps=0):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=collate_fn, drop_last=True)
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        log_step = 50
        total_loss = 0

        for epoch in range(num_epoch):
            start_time = time.time()
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {
                    'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'labels': batch[2],
                    'entity_pos': batch[3],
                    'hts': batch[4],
                }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                    if num_steps % log_step == 0:
                        cur_loss = total_loss / log_step
                        elapsed = time.time() - start_time
                        logging(f"| epoch {epoch} | step {num_steps} | time {elapsed:.2f}s | loss {cur_loss:.4f}")
                        total_loss = 0
                        start_time = time.time()

                if (step + 1) == len(train_dataloader) or (
                    args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0
                ):
                    logging('-' * 89)
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = evaluate(args, model, test_features, tag="test")
                    logging(f"| epoch {epoch} | dev_output: {dev_output} | test_output: {test_output}")
                    if test_score > best_score:
                        best_score = test_score
                        logging(f"Best F1: {best_score}")
                        if args.save_path:
                            torch.save({
                                'epoch': epoch,
                                'checkpoint': model.state_dict(),
                                'best_f1': best_score
                            }, args.save_path)
        return num_steps

    # Optimizer setup
    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'labels': batch[2],
            'entity_pos': batch[3],
            'hts': batch[4],
        }
        with torch.no_grad():
            output = model(**inputs)
            loss = output[0]
            pred = output[1].cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        f"{tag}_F1": f1 * 100,
        f"{tag}_P": precision * 100,
        f"{tag}_R": recall * 100
    }
    return f1, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str)
    parser.add_argument("--train_file", default="train_filter.data", type=str)
    parser.add_argument("--dev_file", default="dev_filter.data", type=str)
    parser.add_argument("--test_file", default="test_filter.data", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--evaluation_steps", default=-1, type=int)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--unet_in_dim", type=int, default=3)
    parser.add_argument("--unet_out_dim", type=int, default=256)
    parser.add_argument("--down_dim", type=int, default=256)
    parser.add_argument("--channel_type", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='')
    parser.add_argument("--bert_lr", default=5e-5, type=float)
    parser.add_argument("--max_height", type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    read = read_cdr if "cdr" in args.data_dir.lower() else read_gda
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features = read(train_file, './train_cache', tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, './dev_cache', tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, './test_cache', tokenizer, max_seq_length=args.max_seq_length)

    bert_model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, args, bert_model, num_labels=args.num_labels)
    model.to(device)

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)
    else:
        train(args, model, train_features, dev_features, test_features)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
