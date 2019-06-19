# -*- coding:utf-8 -*-

import json
import math

import numpy as np
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.predictors.predictor import Predictor


class SLUPredict(Predictor):
    def __init__(self, model, dataset_reader, vocab):
        super(SLUPredict, self).__init__(model, dataset_reader)

        self._domain_vocab = vocab.get_index_to_token_vocabulary('domain_labels')
        self._intent_vocab = vocab.get_index_to_token_vocabulary('intent_labels')
        self._slot_vocab = vocab.get_index_to_token_vocabulary('slots_labels')

    def predict(self, text: str):
        return self.predict_json({"text": text})

    def _json_to_instance(self, json_dict):
        sentence = json_dict["text"]
        return self._dataset_reader.text_to_instance(sentence)

    def predict_instances_to_file(self, instances, path, batch_size=128):
        all_outputs = []
        for i in Tqdm.tqdm(range(math.ceil(len(instances)/batch_size))):
            batch_instances = instances[i*batch_size: (i+1)*batch_size]
            model_outputs = self.predict_batch_instance(batch_instances)

            for j, instance in enumerate(batch_instances):
                outputs = self._decode_by_output(model_outputs[j])
                outputs['text'] = ''.join(map(str, instance.fields['text'].tokens))
                all_outputs.append(outputs)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)

    def _decode_by_output(self, model_outputs):
        outputs = dict()
        outputs['domain'] = self._domain_vocab[np.argmax(model_outputs['domain_probs'])]
        outputs['intent'] = self._intent_vocab[np.argmax(model_outputs['intent_probs'])]

        slots = [self._slot_vocab[np.argmax(element)] for element in model_outputs['slots_probs']]
        outputs['slots'] = dict()
        for catalog, spans in bio_tags_to_spans(slots[: len(outputs['text'])]):
            outputs['slots'][catalog] = outputs['text'][spans[0]: spans[1]+1]
        return outputs
