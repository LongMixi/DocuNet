import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    """
    Long-sequence processing for BERT-like models.
    Supports Transformers 4.x (ModelOutput format).
    """

    n, c = input_ids.size()

    start_tokens = torch.tensor(start_tokens, device=input_ids.device)
    end_tokens = torch.tensor(end_tokens, device=input_ids.device)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)

    # Case 1: short sequence
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        sequence_output = output.last_hidden_state
        attention = output.attentions[-1]   # last layer
        return sequence_output, attention

    # Case 2: long sequence
    new_input_ids = []
    new_attention_mask = []
    num_seg = []

    seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

    for i, l_i in enumerate(seq_len):
        if l_i <= 512:
            new_input_ids.append(input_ids[i, :512])
            new_attention_mask.append(attention_mask[i, :512])
            num_seg.append(1)
        else:
            # left chunk
            ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
            mask1 = attention_mask[i, :512]

            # right chunk
            ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
            mask2 = attention_mask[i, (l_i - 512): l_i]

            new_input_ids.extend([ids1, ids2])
            new_attention_mask.extend([mask1, mask2])
            num_seg.append(2)

    new_input_ids = torch.stack(new_input_ids, dim=0)
    new_attention_mask = torch.stack(new_attention_mask, dim=0)

    output = model(
        input_ids=new_input_ids,
        attention_mask=new_attention_mask,
        output_attentions=True,
        return_dict=True
    )

    sequence_output = output.last_hidden_state
    attention = output.attentions[-1]  # (batch, heads, seq, seq)

    # Now merge chunks back
    i = 0
    new_output, new_attention = [], []

    for n_s, l_i in zip(num_seg, seq_len):

        if n_s == 1:
            out = F.pad(sequence_output[i], (0, 0, 0, c - 512))
            att = F.pad(attention[i], (0, c - 512, 0, c - 512))
            new_output.append(out)
            new_attention.append(att)

        else:  # two chunks
            # First chunk
            out1 = sequence_output[i][:512 - len_end]
            mask1 = new_attention_mask[i][:512 - len_end]
            att1 = attention[i][:, :512 - len_end, :512 - len_end]

            out1 = F.pad(out1, (0, 0, 0, c - 512 + len_end))
            mask1 = F.pad(mask1, (0, c - 512 + len_end))
            att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

            # Second chunk
            out2 = sequence_output[i + 1][len_start:]
            mask2 = new_attention_mask[i + 1][len_start:]
            att2 = attention[i + 1][:, len_start:, len_start:]

            out2 = F.pad(out2, (0, 0, l_i - 512 + len_start, c - l_i))
            mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
            att2 = F.pad(att2, (l_i - 512 + len_start, c - l_i,
                                 l_i - 512 + len_start, c - l_i))

            # merge
            mask = mask1 + mask2 + 1e-10
            out = (out1 + out2) / mask.unsqueeze(-1)

            att = att1 + att2
            att = att / (att.sum(-1, keepdim=True) + 1e-10)

            new_output.append(out)
            new_attention.append(att)

        i += n_s

    sequence_output = torch.stack(new_output, dim=0)
    attention = torch.stack(new_attention, dim=0)

    return sequence_output, attention
