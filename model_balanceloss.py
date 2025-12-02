import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
from losses import balanced_loss as ATLoss
from element_wise import ElementWiseMatrixAttention
from attn_unet import AttentionUNet


# -----------------------------
# PyTorch version of AllenNLP matrix attentions
# -----------------------------
class DotProductMatrix(nn.Module):
    def forward(self, x, y):
        # (bs, n, d) x (bs, d, n) -> (bs, n, n)
        return torch.bmm(x, y.transpose(1, 2))


class CosineMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        # x: (bs, n, d), y: (bs, n, d)
        # output: (bs, n, n)
        bs, n, d = x.size()
        x_exp = x.unsqueeze(2).expand(bs, n, n, d)
        y_exp = y.unsqueeze(1).expand(bs, n, n, d)
        return self.cos(x_exp, y_exp)


class BilinearMatrix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x, y):
        # x W y^T
        # (bs, n, d) * (d, d) -> (bs, n, d)
        out = torch.matmul(x, self.W)          # (bs, n, d)
        return torch.bmm(out, y.transpose(1, 2))  # (bs, n, n)



# ------------------------------------------
# Main DocRE Model (fixed version)
# ------------------------------------------
class DocREModel(nn.Module):
    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()

        self.config = config
        self.bert_model = model
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.bertdrop = nn.Dropout(0.6)

        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_out_dim

        self.liner = nn.Linear(config.hidden_size, args.unet_in_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type

        # Extractor
        self.head_extractor = nn.Linear(config.hidden_size + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(config.hidden_size + args.unet_out_dim, emb_size)

        # Bilinear classifier
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        # U-Net
        self.segmentation_net = AttentionUNet(
            input_channels=args.unet_in_dim,
            class_number=args.unet_out_dim,
            down_channel=args.down_dim
        )

        # Our matrix attentions (AllenNLP replacement)
        self.dot_att = DotProductMatrix()
        self.cos_att = CosineMatrix()
        self.bi_att = BilinearMatrix(self.emb_size)


    # ------------------------------------------
    def encode(self, input_ids, attention_mask, entity_pos):
        config = self.config

        if config.transformer_type in ["bert", "roberta"]:
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        else:
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]

        sequence_output, attention = process_long_input(
            self.bert_model,
            input_ids,
            attention_mask,
            start_tokens,
            end_tokens
        )
        return sequence_output, attention


    # ------------------------------------------
    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1
        bs, heads, _, seq_len = attention.size()

        hss, tss = [], []
        entity_es, entity_as = [], []

        for b in range(len(entity_pos)):
            entity_embs = []
            entity_atts = []

            for e in entity_pos[b]:
                token_embs, token_atts = [], []
                for (start, end) in e:
                    if start + offset < seq_len:
                        token_embs.append(sequence_output[b, start + offset])
                        token_atts.append(attention[b, :, start + offset])

                if len(token_embs) > 0:
                    e_emb = torch.logsumexp(torch.stack(token_embs), dim=0)
                    e_att = torch.stack(token_atts).mean(0)
                else:
                    e_emb = torch.zeros(self.hidden_size, device=sequence_output.device)
                    e_att = torch.zeros(heads, seq_len, device=attention.device)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            # pad to min_height
            while len(entity_atts) < self.min_height:
                entity_atts.append(entity_atts[-1])
                entity_embs.append(entity_embs[-1])

            entity_embs = torch.stack(entity_embs)
            entity_atts  = torch.stack(entity_atts)

            entity_es.append(entity_embs)
            entity_as.append(entity_atts)

            ht_i = torch.LongTensor(hts[b]).to(sequence_output.device)
            hss.append(entity_embs[ht_i[:, 0]])
            tss.append(entity_embs[ht_i[:, 1]])

        hss = torch.cat(hss)
        tss = torch.cat(tss)
        return hss, tss, entity_es, entity_as


    # ------------------------------------------
    def get_ht(self, attn_map, hts):
        outputs = []
        for i in range(len(hts)):
            for (h_i, t_i) in hts[i]:
                outputs.append(attn_map[i, h_i, t_i])
        return torch.stack(outputs)


    # ------------------------------------------
    def get_channel_map(self, sequence_output, entity_as):
        bs, seq_len, dim = sequence_output.size()
        ne = self.min_height

        # Pre-build all index pairs
        idx = torch.stack([
            torch.stack([torch.full((ne,), i, dtype=torch.long),
                         torch.arange(ne)], dim=-1)
            for i in range(ne)
        ], dim=0).reshape(-1, 2).to(sequence_output.device)

        rss = []
        for b in range(bs):
            ent_att = entity_as[b]  # (ne, h, seq)
            h_att = ent_att[idx[:, 0]]
            t_att = ent_att[idx[:, 1]]

            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)

            rs = contract("ld,rl->rd", sequence_output[b], ht_att)
            rss.append(rs)

        rss = torch.stack(rss).reshape(bs, ne, ne, dim)
        return rss


    # ------------------------------------------
    def forward(self, input_ids, attention_mask, labels=None, entity_pos=None, hts=None, instance_mask=None):

        sequence_output, attention = self.encode(input_ids, attention_mask, entity_pos)

        bs, _, d = sequence_output.size()
        ne = max(len(x) for x in entity_pos)

        # entity embeddings
        hs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)

        # Build feature map
        if self.channel_type == "context-based":
            feature_map = self.get_channel_map(sequence_output, entity_as)
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()

        elif self.channel_type == "similarity-based":
            ent_encode = sequence_output.new_zeros(bs, self.min_height, d)
            for b in range(bs):
                e_emb = entity_embs[b]
                ent_encode[b, :e_emb.size(0)] = e_emb

            similar1 = self.dot_att(ent_encode, ent_encode).unsqueeze(-1)
            similar2 = self.cos_att(ent_encode, ent_encode).unsqueeze(-1)
            similar3 = self.bi_att(ent_encode, ent_encode).unsqueeze(-1)

            attn_input = torch.cat([similar1, similar2, similar3], dim=-1)
            attn_input = attn_input.permute(0, 3, 1, 2).contiguous()

        else:
            raise ValueError("Invalid channel_type")

        # UNet
        attn_map = self.segmentation_net(attn_input)
        h_t = self.get_ht(attn_map, hts)

        # final h/t representations
        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))

        # bilinear block
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).reshape(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        pred = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if labels is not None:
            labels = torch.cat([torch.tensor(l, device=logits.device) for l in labels], dim=0)
            loss = self.loss_fnt(logits.float(), labels.float())
            return loss, pred

        return pred
