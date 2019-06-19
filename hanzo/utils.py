# -*- coding:utf-8 -*-

import os
import random


def split_to_train_and_dev(instances, ratio=0.8):
    random.shuffle(instances)

    split_index = int(len(instances) * ratio)
    return instances[: split_index], instances[split_index:]


def create_if_not_exist(path):
    os.makedirs(path, exist_ok=True)
