# -*- coding:utf-8 -*-

import argparse
import os

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import device_mapping
from allennlp.training.trainer import Trainer
from pytorch_pretrained_bert.optimization import BertAdam

from hanzo.datasets import DialogDatasetReader
from hanzo.models import BertForMultiTaskSLU
from hanzo.predicts import SLUPredict
from hanzo.utils import create_if_not_exist, split_to_train_and_dev

VOCAB_DIR = 'vocab'
BEST_MODEL_FILENAME = 'best.th'


def parse_arguments():
    parser = argparse.ArgumentParser(prog='BERT-for-SLU')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')

    parser.add_argument('--dataset', type=str, default='./resources/dialog_dataset/train.json')
    parser.add_argument('--bert', type=str, default='bert-base-chinese')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default='./outputs/default')

    # train args
    train_group = parser.add_argument_group('train')
    train_group.add_argument('--train_ratio', type=float, default=0.8)
    train_group.add_argument('--epoch', type=int, default=10)
    train_group.add_argument('--batch_size', type=int, default=32)
    train_group.add_argument('--lr', type=float, default=5e-5)
    train_group.add_argument('--patience', type=int, help='early stopping')

    # eval args
    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument('--predict_path', type=str, default='./predict.json')

    return parser.parse_args()


opts = parse_arguments()

token_indexer = PretrainedBertIndexer(pretrained_model=opts.bert, max_pieces=64)

reader = DialogDatasetReader(token_indexers={"bert": token_indexer})
dataset = reader.read(opts.dataset)

if opts.train:
    create_if_not_exist(opts.save_dir)

    vocab = Vocabulary.from_instances(dataset)
    vocab.save_to_files(os.path.join(opts.save_dir, VOCAB_DIR))

    model = BertForMultiTaskSLU(vocab, opts.bert)

    if opts.gpu > -1:
        model.cuda(opts.gpu)

    train_dataset, dev_dataset = split_to_train_and_dev(dataset, opts.train_ratio)

    optimizer = BertAdam(model.parameters(), lr=opts.lr)
    iterator = BucketIterator(batch_size=opts.batch_size, sorting_keys=[("text", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      patience=opts.patience,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      validation_metric='+accuracy',
                      cuda_device=opts.gpu,
                      serialization_dir=opts.save_dir,
                      num_epochs=opts.epoch)
    trainer.train()

if opts.eval:
    vocab = Vocabulary.from_files(os.path.join(opts.save_dir, VOCAB_DIR))

    model = BertForMultiTaskSLU(vocab, opts.bert)
    model.load_state_dict(torch.load(os.path.join(opts.save_dir, BEST_MODEL_FILENAME),
                                     map_location=device_mapping(opts.gpu)))

    predictor = SLUPredict(model, reader, vocab)
    predictor.predict_instances_to_file(dataset, opts.predict_path, opts.batch_size)
