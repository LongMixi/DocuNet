import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import json

from model_balanceloss import DocREModel
from utils_sample import set_seed, collate_fn
from evaluation import to_official, official_evaluate
from prepro import ReadDataset


def train(args, model, train_features, dev_features, test_features):
    def logging(s):
        print(s)
        if args.log_dir:
            with open(args.log_dir, 'a+') as f_log:
                f_log.write(s + '\n')

    def finetune(features, optimizer, num_epoch, num_steps):
        cur_model = model.module if hasattr(model, 'module') else model
        best_score = -1
        epoch_delta = 0
        if args.train_from_saved_model:
            checkpoint = torch.load(args.train_from_saved_model)
            best_score = checkpoint["best_f1"]
            epoch_delta = checkpoint["epoch"] + 1

        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=collate_fn, drop_last=True)
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        global_step = 0
        total_loss = 0

        for epoch in range(epoch_delta, epoch_delta + num_epoch):
            start_time = time.time()
            optimizer.zero_grad()
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
                total_loss += loss.item()
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(cur_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            logging(f"| epoch {epoch} | time: {(time.time() - start_time):.2f}s | avg_loss: {total_loss/len(train_dataloader):.4f}")
            total_loss = 0

            # Evaluate
            dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
            logging(f"| epoch {epoch} | dev_score: {dev_score:.4f} | dev_output: {dev_output}")
            if dev_score > best_score:
                best_score = dev_score
                pred = report(args, model, test_features)
                os.makedirs("./submit_result", exist_ok=True)
                with open("./submit_result/best_result.json", "w") as fh:
                    json.dump(pred, fh)
                if args.save_path:
                    torch.save({
                        'epoch': epoch,
                        'checkpoint': cur_model.state_dict(),
                        'best_f1': best_score,
                        'optimizer': optimizer.state_dict()
                    }, args.save_path, _use_new_zipfile_serialization=False)

    cur_model = model.module if hasattr(model, 'module') else model
    extract_layer = ["extractor", "bilinear"]
    bert_layer = ['bert_model']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": args.bert_lr},
        {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
        {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.train_from_saved_model:
        optimizer.load_state_dict(torch.load(args.train_from_saved_model)["optimizer"])

    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps=0)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    preds = []
    total_loss = 0
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
            total_loss += loss.item()
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _, re_p, re_r = official_evaluate(ans, args.data_dir)
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_re_p": re_p * 100,
        tag + "_re_r": re_r * 100,
        tag + "_average_loss": total_loss / len(dataloader)
    }
    return best_f1, output


def report(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    preds = []
    for batch in dataloader:
        model.eval()
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'entity_pos': batch[3],
            'hts': batch[4],
        }
        with torch.no_grad():
            pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    return to_official(preds, features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_labels", default=4, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--bert_lr", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--evaluation_steps", default=-1, type=int)
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--num_class", default=97, type=int)
    parser.add_argument("--unet_in_dim", type=int, default=3)
    parser.add_argument("--unet_out_dim", type=int, default=256)
    parser.add_argument("--down_dim", type=int, default=256)
    parser.add_argument("--channel_type", type=str, default='')
    parser.add_argument("--log_dir", type=str, default='')
    parser.add_argument("--max_height", type=int, default=42)
    parser.add_argument("--train_from_saved_model", type=str, default='')
    parser.add_argument("--dataset", type=str, default='docred')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    Dataset = ReadDataset(args.dataset, tokenizer)

    train_features = Dataset.read(os.path.join(args.data_dir, args.train_file))
    dev_features = Dataset.read(os.path.join(args.data_dir, args.dev_file))
    test_features = Dataset.read(os.path.join(args.data_dir, args.test_file))

    bert_model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, args, bert_model, num_labels=args.num_labels)
    if args.train_from_saved_model:
        model.load_state_dict(torch.load(args.train_from_saved_model)["checkpoint"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path)['checkpoint'])
        pred = report(args, model, test_features)
        os.makedirs("./submit_result", exist_ok=True)
        with open("./submit_result/result.json", "w") as fh:
            json.dump(pred, fh)
    else:
        train(args, model, train_features, dev_features, test_features)


if __name__ == "__main__":
    main()
