from copy import deepcopy
from random import Random

import torch.utils.data


class BertTrainDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, mask_token, num_items, rng: Random, max_len=50, mask_prob=1):
        self.seqs = sequences
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.org_rng = rng
        self.rng = deepcopy(rng)
        self.max_predictions = 40
        self.masked_lm_prob = 0.2
        self.duplicate_factor = 11 # 10/11 are randomly sampled, 1/11 are last item for rec specific loss.

    def __len__(self):
        return len(self.seqs) * self.duplicate_factor

    def __getitem__(self, index):
        seq = self.seqs[index // self.duplicate_factor]

        tokens = list(seq)
        indices = list(range(len(seq)))
        self.rng.shuffle(indices)
        num_to_predict = min(self.max_predictions, max(1, int(round(len(tokens) * self.masked_lm_prob))))
        prob = self.rng.random()

        # 10/11 are randomly sampled, 1/11 are last item for rec specific loss.
        if prob < 0.909:
            n = 0
            for s in indices:
                if n >= num_to_predict:
                    break

                prob = self.rng.random()

                # 80% randomly change token to mask token if self.mask_prob == 0.8
                if prob <= self.mask_prob:
                    tokens[s] = self.mask_token
                else:
                    prob /= self.mask_prob  # normalize to [0, 1]
                    if prob < 0.5:
                        # 10% randomly change token to random token
                        tokens[s] = self.rng.randint(0, self.num_items)
                    else:
                        # 10% randomly change token to current token
                        tokens[s] = seq[s]

                n += 1

            positions = indices[:n]
        else:
            tokens[-1] = self.mask_token
            positions = [len(seq) - 1]

        # revert order
        tokens = tokens[-self.max_len:]
        labels = [-1] * len(tokens)

        for i in positions:
            labels[i] = seq[i]

        seq_mask_len = self.max_len - len(tokens)
        # label_mask_len = self.max_predictions - len(labels)

        tokens += [-1] * seq_mask_len
        labels += [-1] * seq_mask_len

        return torch.LongTensor(tokens) + 1, torch.LongTensor(labels)

    def reset_rng(self):
        self.rng = deepcopy(self.org_rng)