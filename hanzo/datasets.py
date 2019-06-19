# -*- coding:utf-8 -*-

import json
from typing import Dict, Iterable, List

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, SequenceLabelField, TextField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token


class DialogDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]):
        """
        dataset example:
        [
          {
            "text": "老马的电话是多少",
            "domain": "telephone",
            "intent": "QUERY",
            "slots": {
              "name": "老马"
            }
          }
        ]
        """
        super(DialogDatasetReader, self).__init__(lazy=False)

        if 'bert' in token_indexers:
            # deal with space in bert, because of the case: `打电话给my god`
            wordpiece_tokenizer = token_indexers['bert'].wordpiece_tokenizer
            token_indexers['bert'].wordpiece_tokenizer = \
                lambda s: ['[UNK]'] if s.isspace() else wordpiece_tokenizer(s)
        self.token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)

        for instance in data:
            self._preprocess_labels(instance)
            text = instance['text']
            domain = instance.get('domain')
            intent = instance.get('intent')
            slots = instance.get('slots')

            yield self.text_to_instance(text, domain, intent, slots)

    def text_to_instance(self, text: str, domain: str = None, intent: str = None,
                         slots: List[str] = None) -> Instance:
        fields = dict()
        fields['text'] = TextField([Token(word) for word in text], self.token_indexers)
        if domain:
            fields['domain'] = LabelField(domain, label_namespace='domain_labels')
            fields['intent'] = LabelField(intent, label_namespace='intent_labels')
            fields['slots'] = SequenceLabelField(labels=slots,
                                                 sequence_field=fields['text'],
                                                 label_namespace='slots_labels')
        return Instance(fields)

    @staticmethod
    def _preprocess_labels(instance):
        raw_text = instance['text']
        raw_slots = instance.get('slots')

        if not instance.get('domain'):  # check whether it is train mod
            return

        # process intent
        intent = instance.get('intent')
        if isinstance(intent, float):
            instance['intent'] = str(instance['intent'])

        # process slots
        slots = ['O'] * len(instance['text'])
        for slot_type, slot_span in raw_slots.items():
            start = raw_text.index(slot_span)
            end = start + len(slot_span)
            slots[start: end] = ['B-{}'.format(slot_type)] + \
                                ['I-{}'.format(slot_type) for _ in range(len(slot_span) - 1)]
        instance['slots'] = slots
