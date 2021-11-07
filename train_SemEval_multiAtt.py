# coding:utf-8
import sys, json
import torch
import os
import numpy as np
import argparse
import logging
import random
from multiAtt_encoder import MultiAttEnocder
from multiAtt_nn import MultiAtt_NN
from multiAtt_re import MultiAttRE
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=os.environ.get("LOGLEVEL", "INFO"))
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed = 42
ckpt = ''
dataset = 'SemEval2010-Task8'
train_file = ''
val_file = ''
test_file = ''
rel2id_file = ''
max_length = 128
lr = 0.03
batch_size = 256
max_epoch = 500
weight_decay = 0.0001
optim = 'adam'
only_test = False
metric = 'macro_f1'
conv_size = 1000

# Set random seed
set_seed(seed)

# Some basic settings
root_path = '.'
sys.path.append(root_path) #add to the search path
if not os.path.exists('ckpt'): #the directory to save the model
    os.mkdir('ckpt')
if len(ckpt) == 0:
    ckpt = '{}_{}'.format(dataset, 'multiAtt')
ckpt = 'ckpt/{}.pth.tar'.format(ckpt)


if dataset != 'none':
    train_file = os.path.join(root_path, 'data', dataset, '{}_train.txt'.format(dataset))
    val_file = os.path.join(root_path, 'data', dataset, '{}_val.txt'.format(dataset))
    if not os.path.exists(val_file):
        logging.info("Cannot find the validation file. Use the test file instead.")
        val_file = os.path.join(root_path, 'data', dataset, '{}_test.txt'.format(dataset))
    test_file = os.path.join(root_path, 'data', dataset, '{}_test.txt'.format(dataset))
    rel2id_file = os.path.join(root_path, 'data', dataset, '{}_rel2id.json'.format(dataset))
else:
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file) and os.path.exists(rel2id_file)):
        raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

rel2id = json.load(open(rel2id_file))

word2id = json.load(open(os.path.join(root_path, 'data/glove/glove.6B.50d_word2id.json')))
word2vec = np.load(os.path.join(root_path, 'data/glove/glove.6B.50d_mat.npy'))

sentence_encoder = MultiAttEnocder(
        token2id=word2id,
        max_length=max_length,
        word_size=50,
        position_size=5,
        hidden_size=conv_size,
        blank_padding=True, #When the sentence length is not long enough, fill in the blanks
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )

model = MultiAtt_NN(sentence_encoder,len(rel2id),rel2id)

framework = MultiAttRE(model,train_file,val_file,test_file,ckpt,batch_size=batch_size,max_epoch=max_epoch)

# Train the model
if not only_test:
    framework.train_model(metric)

