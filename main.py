import os
import datetime
import argparse
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.bert.modeling_bert import (
    BertLayer,
    BertEmbeddings,
    BertPooler,
    BertEncoder,
)
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from utils.buffer import TensorBuffer

list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],]

class TestClass:
    
    def __init__(self, list):
        self.list = list[::-1]
    
    def get_list(self):
        for i in range(len(self.list)):
            for j in range(len(self.list[i])):
                self.list[i].append(self.list[i][j] * 2)
            
test_class = TestClass(list)
test_class.get_list()

print(list)