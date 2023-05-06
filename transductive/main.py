# coding=UTF-8
import os
import argparse
from tqdm import tqdm
import torch
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from trainer import Trainer
from dataloader import DataLoader
from model import Model
from datetime import datetime
import logging


def set_logger(data):
	'''
	Write logs to checkpoint and console
	'''
	now = datetime.now()
	save_folder = './outputs/'

	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	log_file = save_folder + data + now.strftime("%Y%m%d%H%M%S") + '.log'

	logging.basicConfig(
		format='%(asctime)s %(levelname)-8s %(message)s',
		level=logging.INFO,
		datefmt='%Y-%m-%d %H:%M:%S',
		filename=log_file,
		filemode='w'
	)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


if __name__ == '__main__':
	# argparser
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	
	parser.add_argument('--bert_lr', type=float, default=1e-5)
	parser.add_argument('--model_lr', type=float, default=5e-4)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--epoch', type=int, default=100)
	parser.add_argument('--weight_decay', type=float, default=1e-6)
	parser.add_argument('--data', type=str, default='fb15k-237') 
	parser.add_argument('--plm', type=str, default='bert', choices = ['bert', 'bert_tiny', 'deberta'])
	parser.add_argument('--self_adversarial', default=False, action = 'store_true', help = 'self adversarial negative sampling')
	parser.add_argument('--model', type=str, default='NTRL', choices = ['NTRL', 'BLP', 'DKRL'])
	parser.add_argument('--text_type', type=str, default='neighbor_text', choices=['neighbor_text', 'desc_text', 'con_text'])
	parser.add_argument('--num_facts', type=int, default=0, help = 'the number of entity first-order neighborhood facts')
	parser.add_argument('--num_tokens', type=int, default=50, help='the number of entity description text tokens')
	parser.add_argument('--score_function', type=str, default='cross_mln', choices = ['transe', 'distmult', 'complex','simple','mln','cross_mln'])
	parser.add_argument('--hr', default=False, action='store_true', help='hr cross feature')
	parser.add_argument('--tr', default=False, action='store_true', help='tr cross feature')
	parser.add_argument('--ht', default=False, action='store_true', help='ht cross feature')
	parser.add_argument('--hrt', default=False, action='store_true', help='hrt cross feature')
	parser.add_argument('--lamda', type=str,default='lamda1', choices = ['lamda1', 'lamda2'])
	arg = parser.parse_args()


	# Set random seed
	random.seed(arg.seed)
	np.random.seed(arg.seed)
	torch.manual_seed(arg.seed)

	device = torch.device('cuda')

	if arg.plm == 'bert':
		plm_name = "bert-base-uncased"
		t_model = 'bert'
	elif arg.plm == 'bert_tiny':
		plm_name = "prajjwal1/bert-tiny"
		t_model = 'bert'
	elif arg.plm =='deberta':
		plm_name = 'microsoft/deberta-base'
		t_model = 'bert'

	if arg.data == 'umls':
		in_paths = {
			'dataset': arg.data,
			'train': './data/umls/train.tsv',
			'valid': './data/umls/dev.tsv',
			'test': './data/umls/test.tsv',
			'text': ['./data/umls/entity2text.txt', './data/umls/relation2text.txt', './data/umls/entity2textlong.txt']
		}
	elif arg.data == 'fb15k-237':
		in_paths = {
			'dataset': arg.data,
			'train': './data/fb15k-237/train.tsv',
			'valid': './data/fb15k-237/dev.tsv',
			'test': './data/fb15k-237/test.tsv',
			'text': ['./data/fb15k-237/entity2text.txt',
				'./data/fb15k-237/relation2name.txt',
				'./data/fb15k-237/entity2textlong.txt']
		}
	elif arg.data == 'wn18rr':
		in_paths = {
			'dataset': arg.data,
			'train': './data/WN18RR/train.tsv',
			'valid': './data/WN18RR/dev.tsv',
			'test': './data/WN18RR/test.tsv',
			'text': ['./data/WN18RR/entity2name.txt',
				'./data/WN18RR/relation2text.txt',
				'./data/WN18RR/entity2text.txt']
		}

	lm_config = AutoConfig.from_pretrained(plm_name, cache_dir = '../cached_model',block_size=4, num_random_blocks=1)
	lm_tokenizer = AutoTokenizer.from_pretrained(plm_name, do_basic_tokenize=False, cache_dir = '../cached_model')
	lm_model = AutoModel.from_pretrained(plm_name, config=lm_config, cache_dir = '../cached_model',)
	
	

	data_loader = DataLoader(in_paths, lm_tokenizer, arg.batch_size,arg.text_type,arg.num_facts,arg.num_tokens, t_model)


	model = Model(lm_model, len(data_loader.ent2id), len(data_loader.rel2id) , arg.model ,arg.score_function,arg.hr,arg.tr,arg.ht,arg.hrt)

	no_decay = ["bias", "LayerNorm.weight"]
	param_group = [
		{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
									   if ('lm_model' not in n) and
									   (not any(nd in n for nd in no_decay))],
		 'weight_decay': arg.weight_decay},
		{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
									   if ('lm_model' not in n) and
									   (any(nd in n for nd in no_decay))],
		 'weight_decay': 0.0},
	]
	param_group += [
		{'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
									   if ('lm_model' in n) and
									   (not any(nd in n for nd in no_decay)) ], # name中不包含bias和LayerNorm.weight
		 'weight_decay': arg.weight_decay},
		{'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
									   if ('lm_model' in n) and
									   (any(nd in n for nd in no_decay))],
		 'weight_decay': 0.0},
	]

	optimizer = AdamW(param_group) # transformer AdamW

	hyperparams = {
		'batch_size': arg.batch_size,
		'epoch': arg.epoch,
		'evaluate_every': 1, 
		'update_every': 1,
		'plm': arg.plm,
		'self_adversarial':arg.self_adversarial,
		'data': arg.data,
		'model':arg.model,
		'text_type':arg.text_type,
		'num_facts':arg.num_facts,
		'lamda': arg.lamda
	}
	set_logger(arg.data)
	logging.info('Model: %s' % arg.model)
	logging.info('seed: %d' % arg.seed)
	logging.info('Data: %s' % arg.data)
	logging.info('num_facts: %d' % arg.num_facts)
	logging.info('bert_lr: %f' % arg.bert_lr)
	logging.info('model_lr: %f' % arg.model_lr)
	logging.info('epoch: %d' % arg.epoch)
	logging.info('weight_decay: %f' % arg.weight_decay)
	logging.info('plm: %s' % arg.plm)
	logging.info('self_adversarial: %s' % arg.self_adversarial)
	logging.info('text_type: %s' % arg.text_type)
	logging.info('num_tokens: %d' % arg.num_tokens)
	logging.info('score_function: %s' % arg.score_function)
	logging.info('hr: %s' % arg.hr)
	logging.info('tr: %s' % arg.tr)
	logging.info('ht: %s' % arg.ht)
	logging.info('hrt: %s' % arg.hrt)
	logging.info('lamda: %s' % arg.lamda)

	trainer = Trainer(data_loader, model, lm_tokenizer, optimizer, device, hyperparams)
	trainer.run()


