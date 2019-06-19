# -*- coding:utf-8 -*-

from typing import Dict, Union

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import \
    PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure
from pytorch_pretrained_bert.modeling import BertModel


class BertForTagger(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: Union[str, BertModel],
                 dropout: float = 0.0,
                 num_labels: int = None,
                 index: str = "bert",
                 label_namespace: str = "labels",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model
        self.bert_model.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size
        if num_labels:
            out_features = num_labels
        else:
            out_features = vocab.get_vocab_size(label_namespace)

        self._dropout = torch.nn.Dropout(p=dropout)
        self._tagger_layer = torch.nn.Linear(in_features, out_features)
        self._span_f1 = SpanBasedF1Measure(vocab, label_namespace, label_encoding='BIO')
        self._loss = torch.nn.CrossEntropyLoss()
        self._index = index
        initializer(self._tagger_layer)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        sequence_output, _ = self.bert_model(input_ids=input_ids,
                                             token_type_ids=token_type_ids,
                                             attention_mask=input_mask,
                                             output_all_encoded_layers=False)

        # delete [CLS] and [SEQ]
        sequence_output = sequence_output[:, 1:-1]
        input_mask = input_mask[:, 1:-1]
        sequence_output = self._dropout(sequence_output)

        # apply tagger layer
        tag_logits = self._tagger_layer(sequence_output)
        tag_probs = torch.nn.functional.softmax(tag_logits, dim=-1)
        output_dict = {'logits': tag_logits, 'probs': tag_probs}

        if labels is not None:
            loss = sequence_cross_entropy_with_logits(tag_logits, labels, input_mask)
            output_dict["loss"] = loss
            self._span_f1(tag_logits, labels, input_mask)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"f1": self._span_f1.get_metric(reset)['f1-measure-overall']}
