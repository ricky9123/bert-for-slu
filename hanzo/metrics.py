# -*- coding:utf-8 -*-

import torch
from allennlp.training.metrics.metric import Metric


class SentenceAccuracy(Metric):
    def __init__(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self, domain_pred=None, domain_gold=None,
                 intent_pred=None, intent_gold=None,
                 slots_pred=None, slots_gold=None, slots_mask=None):
        domain_pred, domain_gold, intent_pred, intent_gold, \
            slots_pred, slots_gold, slots_mask = \
            self.unwrap_to_tensors(domain_pred, domain_gold, intent_pred, intent_gold,
                                   slots_pred, slots_gold, slots_mask)

        domain_res = (torch.argmax(domain_pred, dim=-1) == domain_gold)
        intent_res = (torch.argmax(intent_pred, dim=-1) == intent_gold)
        slots_res = (torch.argmax(slots_pred, dim=-1) * slots_mask == slots_gold * slots_mask)
        slots_res = torch.all(slots_res, dim=-1)

        overall_res = domain_res * intent_res * slots_res  # all is True
        self.correct_count += overall_res.sum()
        self.total_count += overall_res.numel()

    def get_metric(self, reset: bool):
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
