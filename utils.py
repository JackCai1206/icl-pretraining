import torch

def align_sentences(sent_ids, sep_id, pad_id):
    B, L = sent_ids.shape
    sep_pos = torch.argmax((sent_ids == sep_id).cumsum(1), 1) # find the last separator token
    assert sep_pos.shape[0] == B, 'All inputs must have exactly one separator token'
    sent_start = torch.max(sent_ids != pad_id, dim=1).indices
    sent_len_temp = sep_pos.max() - sep_pos.min() + L
    sent_len = (sep_pos - sent_start).max() + (L - sep_pos).max()
    aligned_ids = torch.full((B, sent_len_temp), pad_id)
    start_pos = sep_pos.max() - sep_pos
    aligned_ids[:, :L] = sent_ids
    roll_indices = (torch.arange(aligned_ids.shape[1])[None, :] - start_pos[:, None]) % aligned_ids.shape[1]
    aligned_ids = torch.gather(aligned_ids, 1, roll_indices)
    if sent_len_temp != sent_len: assert (aligned_ids[:, 0] == pad_id).all()
    aligned_ids = aligned_ids[:, -sent_len:]
    sep_pos = sep_pos.max() - (sent_len_temp - sent_len)

    return aligned_ids, sep_pos
