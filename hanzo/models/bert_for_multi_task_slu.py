# -*- coding:utf-8 -*-

from typing import Dict, Union

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.bert_for_classification import BertForClassification
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import \
    PretrainedBertModel
from pytorch_pretrained_bert.modeling import BertModel

from hanzo.metrics import SentenceAccuracy
from hanzo.models.bert_for_tagger import BertForTagger


class BertForMultiTaskSLU(Model):
    def __init__(self, vocab: Vocabulary, bert_model: Union[str, BertModel]):
        super().__init__(vocab)

        bert_model = PretrainedBertModel.load(bert_model)
        self.bert_for_domain_classification = BertForClassification(vocab, bert_model, label_namespace='domain_labels')
        self.bert_for_intent_classification = BertForClassification(vocab, bert_model, label_namespace='intent_labels')
        self.bert_for_slot_filling = BertForTagger(vocab, bert_model, label_namespace='slots_labels')

        self._accuracy = SentenceAccuracy()

    def forward(self,
                text: Dict[str, torch.LongTensor],
                domain: torch.Tensor = None,
                intent: torch.Tensor = None,
                slots: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        domain_outputs = self.bert_for_domain_classification.forward(text, domain)
        intent_outputs = self.bert_for_intent_classification.forward(text, intent)
        slots_outputs = self.bert_for_slot_filling.forward(text, slots)

        output_dict = {'domain_probs': domain_outputs['probs'],
                       'intent_probs': intent_outputs['probs'],
                       'slots_probs': slots_outputs['probs']}

        if domain is not None and intent is not None and slots is not None:
            output_dict['loss'] = domain_outputs['loss'] + intent_outputs['loss'] + slots_outputs['loss']
            self._accuracy(output_dict['domain_probs'], domain,
                           output_dict['intent_probs'], intent,
                           output_dict['slots_probs'], slots, text['mask'])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        domain_metrics = self.bert_for_domain_classification.get_metrics(reset)
        intent_metrics = self.bert_for_intent_classification.get_metrics(reset)
        slots_metrics = self.bert_for_slot_filling.get_metrics(reset)

        metrics = dict()
        metrics['domain_accuracy'] = domain_metrics['accuracy']
        metrics['intent_accuracy'] = intent_metrics['accuracy']
        metrics['slots_f1'] = slots_metrics['f1']
        metrics['accuracy'] = self._accuracy.get_metric(reset)

        return metrics
