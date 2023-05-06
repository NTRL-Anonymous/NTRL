import os
import random
import math
import pickle
import torch
import time 
from tqdm import tqdm
import copy
from transformers import BatchEncoding
from collections import defaultdict
import numpy as np

class DataSampler(object):
	def __init__(self, datasetName, mode, pos_dataset, whole_dataset, batch_size, entity_set, relation_set, groundtruth=None, possible_entities=None):
		self.datasetName = datasetName

		self.batch_size = batch_size
		self.entity_set = entity_set
		self.relation_set = relation_set

		self.mode = mode
		self.whole_dataset = whole_dataset

		self.groundtruth = groundtruth
		self.possible_entities = possible_entities
		self.dataset = self.create_dataset(pos_dataset)

		self.n_batch = math.ceil(len(self.dataset) / batch_size)
		self.i_batch = 0

	def create_dataset(self, pos_dataset):
		dataset = [] 
		random.shuffle(pos_dataset)
		pos_dataset_set = set(pos_dataset)
		whole_dataset_set = set(self.whole_dataset)

		for triple in tqdm(pos_dataset,desc = 'create_dataset'):
			dataset.append((triple, 1))
			h, r, t = triple
		return dataset

	def __iter__(self):
		return self 

	def __next__(self):
		if self.i_batch == self.n_batch:
			raise StopIteration()

		batch = self.dataset[self.i_batch*self.batch_size: (self.i_batch+1)*self.batch_size]
			
		self.i_batch += 1
		return batch

	def __len__(self):
		return self.n_batch

	def get_dataset_size(self):
		return len(self.dataset)


class DataLoader(object):
	def __init__(self, in_paths, tokenizer, batch_size,text_type,num_facts,num_tokens, model='bert'):
		
		self.datasetName = in_paths['dataset'] 

		self.train_set = self.load_dataset(in_paths['train'])
		self.valid_set = self.load_dataset(in_paths['valid'])
		self.test_set = self.load_dataset(in_paths['test'])
		self.valid_set_with_neg = None
		self.test_set_with_neg = None


		self.whole_set = self.train_set + self.valid_set + self.test_set

		self.uid2text =  {}
		self.uid2tokens =  {}
		self.uid2longtext = {}
		self.uid2longtokens = {}

		self.entity_set = set([t[0] for t in (self.train_set + self.valid_set + self.test_set)] + [t[-1] for t in (self.train_set + self.valid_set + self.test_set)])
		self.relation_set = set([t[1] for t in (self.train_set + self.valid_set + self.test_set)])

		self.tokenizer = tokenizer
		for p in in_paths['text'][:2]:
			self.load_text(p)
		for p in in_paths['text'][2:]:
			self.load_longtext(p)
		self.ent2tris(self.train_set)


		self.batch_size = batch_size
		self.text_type = text_type
		self.num_facts = num_facts
		self.num_tokens = num_tokens
		self.step_per_epc = math.ceil(len(self.train_set) / batch_size)


		self.train_entity_set = set([t[0] for t in self.train_set] + [t[-1] for t in self.train_set])
		self.train_relation_set = set([t[1] for t in self.train_set])


		self.entity_list = sorted(self.entity_set)
		self.relation_list = sorted(self.relation_set)

		self.ent2id = {e:i for i, e in enumerate(sorted(self.entity_set))}
		self.rel2id = {r:i for i, r in enumerate(sorted(self.relation_set))}
		self.groundtruth, self.possible_entities= self.count_groundtruth()
		self.model = model
		self.orig_vocab_size = len(tokenizer)

	def ent2tris(self, train_set):
		ent2tris_set = defaultdict(set)
		for h,r,t in train_set:
			ent2tris_set[h].add((h,r,t))
			ent2tris_set[t].add((h,r,t))
		ent2tris_list = defaultdict(list)
		for key,value in ent2tris_set.items():
			ent2tris_list[key] = list(value)

		self.ent2tris = ent2tris_list

	def load_dataset(self, in_path):
		dataset = []
		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				if in_path[-3:] == 'txt':
					r, h, t = line.strip('\n').split('\t')
				else:
					h, r, t = line.strip('\n').split('\t')
				dataset.append((h, r, t))
		return dataset

	def load_text(self, in_path):
		uid2text = self.uid2text
		uid2tokens = self.uid2tokens

		tokenizer = self.tokenizer


		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				uid, text = line.strip('\n').split('\t', 1)
				text = text.replace('@en', '').strip('"')
				if uid not in uid2text.keys():
					uid2text[uid] = text

				tokens = tokenizer.tokenize(text)

				if uid not in uid2tokens.keys():
					uid2tokens[uid] = tokens
		self.uid2text = uid2text
		self.uid2tokens = uid2tokens

	def load_longtext(self, in_path):
		uid2longtext = self.uid2longtext
		uid2longtokens = self.uid2longtokens
		tokenizer = self.tokenizer

		with open(in_path, 'r', encoding='utf8') as fil:
			for line in fil.readlines():
				uid, text = line.strip('\n').split('\t', 1)
				text = text.replace('@en', '').strip('"')
				if uid not in uid2longtext.keys():
					uid2longtext[uid] = text

				tokens = tokenizer.tokenize(text)

				if uid not in uid2longtokens.keys():
					uid2longtokens[uid] = tokens

		self.uid2longtext = uid2longtext
		self.uid2longtokens = uid2longtokens

	def triple_to_text(self, triple):

		tokenizer = self.tokenizer
		ent2id = self.ent2id
		rel2id = self.rel2id

		h, r, t = triple
		h_tris = self.ent2tris[h][:self.num_facts]
		t_tris = self.ent2tris[t][:self.num_facts]
		if triple in h_tris:
			h_tris.remove(triple)
		if triple in t_tris:
			t_tris.remove(triple)
		h_tris_h = defaultdict(list)
		h_tris_t = defaultdict(list)
		h_tris_r_h = []
		h_tris_r_t = []
		for tr in h_tris:
			if tr[0] == h:
				h_tris_h[tr[1]].append(tr)
				h_tris_r_h.append(tr[1])
			else:
				h_tris_t[tr[1]].append(tr)
				h_tris_r_t.append(tr[1])
		h_tris_r_h = list(set(h_tris_r_h))
		h_tris_r_t = list(set(h_tris_r_t))
		h_tris_r_h.sort()
		h_tris_r_t.sort()

		t_tris_h = defaultdict(list)
		t_tris_t = defaultdict(list)
		t_tris_r_h = []
		t_tris_r_t = []
		for tr in t_tris:
			if tr[0] == t:
				t_tris_h[tr[1]].append(tr)
				t_tris_r_h.append(tr[1])
			else:
				t_tris_t[tr[1]].append(tr)
				t_tris_r_t.append(tr[1])
		t_tris_r_h = list(set(t_tris_r_h))
		t_tris_r_t = list(set(t_tris_r_t))
		t_tris_r_h.sort()
		t_tris_r_t.sort()
		if self.model == 'roberta':
			start_tokens = ['<s>']
			end_tokens = ['</s>']
		else:
			start_tokens = ['[CLS]']
			end_tokens = ['[SEP]']

		h_text_tokens = start_tokens.copy()
		types_h = [0]
		if self.text_type == 'desc_text' or self.text_type == 'con_text':
			h_text_tokens.extend(self.uid2longtokens.get(h, [])[:self.num_tokens])
			types_h.extend([0] * len(self.uid2longtokens.get(h, [])[:self.num_tokens]))
			h_text_tokens.extend(end_tokens)
			types_h.extend([0])
		if self.text_type == 'neighbor_text' or self.text_type == 'con_text':
			h_text_tokens.extend(self.uid2tokens.get(h, []))
			types_h.extend([0] * len(self.uid2tokens.get(h, [])))
			h_text_tokens.extend(end_tokens)
			types_h.extend([0])
			for rel in h_tris_r_h:
				h_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_h.extend([1] * len(self.uid2tokens.get(rel, [])))
				for tri in h_tris_h[rel]:
					h1 ,r1 ,t1 = tri
					h_text_tokens.extend(self.uid2tokens.get(t1, []))
					types_h.extend([1] * len(self.uid2tokens.get(t1, [])))
					h_text_tokens.extend([','])
					types_h.extend([1])
				types_h.pop()
				h_text_tokens.pop()
				h_text_tokens.extend(end_tokens)
				types_h.extend([1])
			for rel in h_tris_r_t:
				for tri in h_tris_t[rel]:
					h1, r1, t1 = tri
					h_text_tokens.extend(self.uid2tokens.get(h1, []))
					types_h.extend([1] * len(self.uid2tokens.get(h1, [])))
					h_text_tokens.extend([','])
					types_h.extend([1])
				h_text_tokens.pop()
				types_h.pop()
				h_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_h.extend([1] * len(self.uid2tokens.get(rel, [])))
				h_text_tokens.extend(end_tokens)
				types_h.extend([1])

		t_text_tokens = start_tokens.copy()
		types_t = [0]
		if self.text_type == 'desc_text' or self.text_type == 'con_text':
			t_text_tokens.extend(self.uid2longtokens.get(t, [])[:self.num_tokens])
			types_t.extend([0] * len(self.uid2longtokens.get(t, [])[:self.num_tokens]))
			t_text_tokens.extend(end_tokens)
			types_t.extend([0])
		if self.text_type == 'neighbor_text' or self.text_type == 'con_text':
			t_text_tokens.extend(self.uid2tokens.get(t, []))
			types_t.extend([0] * len(self.uid2tokens.get(t, [])))
			t_text_tokens.extend(end_tokens)
			types_t.extend([0])
			for rel in t_tris_r_h:
				t_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_t.extend([1] * len(self.uid2tokens.get(rel, [])))
				for tri in t_tris_h[rel]:
					h1, r1, t1 = tri
					t_text_tokens.extend(self.uid2tokens.get(t1, []))
					types_t.extend([1] * len(self.uid2tokens.get(t1, [])))
					t_text_tokens.extend([','])
					types_t.extend([1])
				types_t.pop()
				t_text_tokens.pop()
				t_text_tokens.extend(end_tokens)
				types_t.extend([1])
			for rel in t_tris_r_t:
				for tri in t_tris_t[rel]:
					h1, r1, t1 = tri
					t_text_tokens.extend(self.uid2tokens.get(h1, []))
					types_t.extend([1] * len(self.uid2tokens.get(h1, [])))
					t_text_tokens.extend([','])
					types_t.extend([1])
				t_text_tokens.pop()
				types_t.pop()
				t_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_t.extend([1] * len(self.uid2tokens.get(rel, [])))
				t_text_tokens.extend(end_tokens)
				types_t.extend([1])

		text_h = tokenizer.convert_tokens_to_string(h_text_tokens)
		text_t = tokenizer.convert_tokens_to_string(t_text_tokens)

		return text_h,text_t, h_text_tokens,t_text_tokens,types_h,types_t

	def element_to_text(self, target):
		tokenizer = self.tokenizer
		ent2id = self.ent2id
		rel2id = self.rel2id

		if self.model == 'roberta':
			start_tokens = ['<s>']
			end_tokens = ['</s>']
		else:
			start_tokens = ['[CLS]']
			end_tokens = ['[SEP]']

		target_tris = self.ent2tris[target][:self.num_facts]
		target_tris_h = defaultdict(list)
		target_tris_t = defaultdict(list)
		target_tris_r_h = []
		target_tris_r_t = []
		for tr in target_tris:
			if tr[0] == target:
				target_tris_h[tr[1]].append(tr)
				target_tris_r_h.append(tr[1])
			else:
				target_tris_t[tr[1]].append(tr)
				target_tris_r_t.append(tr[1])
		target_tris_r_h = list(set(target_tris_r_h))
		target_tris_r_t = list(set(target_tris_r_t))
		target_tris_r_h.sort()
		target_tris_r_t.sort()
		types_target = [0]
		target_text_tokens = start_tokens.copy()
		if self.text_type == 'desc_text' or self.text_type == 'con_text':
			target_text_tokens.extend(self.uid2longtokens.get(target, [])[:self.num_tokens])
			types_target.extend([0] * len(self.uid2longtokens.get(target, [])[:self.num_tokens]))
			target_text_tokens.extend(end_tokens)
			types_target.extend([0])
		if self.text_type == 'neighbor_text' or self.text_type == 'con_text':
			target_text_tokens.extend(self.uid2tokens.get(target, []))
			types_target.extend([0] * len(self.uid2tokens.get(target, [])))
			target_text_tokens.extend(end_tokens)
			types_target.extend([0])
			for rel in target_tris_r_h:
				target_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_target.extend([1] * len(self.uid2tokens.get(rel, [])))
				for tri in target_tris_h[rel]:
					h1, r1, t1 = tri
					target_text_tokens.extend(self.uid2tokens.get(t1, []))
					types_target.extend([1] * len(self.uid2tokens.get(t1, [])))
					target_text_tokens.extend([','])
					types_target.extend([1])
				types_target.pop()
				target_text_tokens.pop()
				target_text_tokens.extend(end_tokens)
				types_target.extend([1])
			for rel in target_tris_r_t:
				for tri in target_tris_t[rel]:
					h1, r1, t1 = tri
					target_text_tokens.extend(self.uid2tokens.get(h1, []))
					types_target.extend([1] * len(self.uid2tokens.get(h1, [])))
					target_text_tokens.extend([','])
					types_target.extend([1])
				target_text_tokens.pop()
				types_target.pop()
				target_text_tokens.extend(self.uid2tokens.get(rel, []))
				types_target.extend([1] * len(self.uid2tokens.get(rel, [])))
				target_text_tokens.extend(end_tokens)
				types_target.extend([1])
		text_target = tokenizer.convert_tokens_to_string(target_text_tokens)
		return text_target, target_text_tokens,types_target

	def batch_tokenize(self, batch_triples):
		batch_texts_h = []
		batch_texts_t = []

		batch_tokens_h = []
		batch_tokens_t = []

		batch_types_h = []
		batch_types_t = []

		batch_positions = []

		ent2id = self.ent2id
		rel2id = self.rel2id


		for triple in batch_triples:
			text_h , text_t, h_text_tokens,t_text_tokens,types_h,types_t = self.triple_to_text(triple)
			batch_texts_h.append(text_h)
			batch_texts_t.append(text_t)
			batch_tokens_h.append(h_text_tokens)
			batch_tokens_t.append(t_text_tokens)
			batch_types_h.append(types_h)
			batch_types_t.append(types_t)

			h, r, t = triple


		batch_tokens_h = self.my_tokenize(batch_tokens_h,batch_types_h, max_length=512, padding=True, model=self.model)
		batch_tokens_t = self.my_tokenize(batch_tokens_t,batch_types_t, max_length=512, padding=True, model=self.model)
		return batch_tokens_h,batch_tokens_t


	def batch_tokenize_target(self, targets = None):
		batch_texts = []
		batch_tokens = []
		batch_positions = []
		batch_types = []
		ent2id = self.ent2id
		rel2id = self.rel2id

		for target in targets:
			text, tokens,types = self.element_to_text(target)
			batch_texts.append(text)
			batch_tokens.append(tokens)
			batch_types.append(types)
		batch_tokens = self.my_tokenize(batch_tokens,batch_types, max_length=512, padding=True, model=self.model)


		return batch_tokens

	def train_data_sampler(self):
		return DataSampler(datasetName = self.datasetName, mode='train', pos_dataset=self.train_set, whole_dataset=self.whole_set, batch_size=self.batch_size, 
			entity_set=self.train_entity_set, relation_set=self.train_relation_set, groundtruth=self.groundtruth,
			possible_entities=self.possible_entities)

	def get_dataset_size(self, split='train'):
		if split == 'train':
			return len(self.train_set)

	def count_groundtruth(self):
		groundtruth = { split: {'head': {}, 'rel': {}, 'tail': {}} for split in ['all', 'train', 'valid', 'test']}
		possible_entities = { split: {'head': {}, 'tail': {}} for split in ['train']}

		for triple in self.train_set:
			h, r, t = triple
			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)  
			groundtruth['train']['head'].setdefault((r, t), [])
			groundtruth['train']['head'][(r, t)].append(h)
			groundtruth['train']['tail'].setdefault((r, h), [])
			groundtruth['train']['tail'][(r, h)].append(t) 
			groundtruth['train']['rel'].setdefault((h, t), [])
			groundtruth['train']['rel'][(h, t)].append(r) 
			possible_entities['train']['head'].setdefault(r, set())
			possible_entities['train']['head'][r].add(h)
			possible_entities['train']['tail'].setdefault(r, set())
			possible_entities['train']['tail'][r].add(t)
		

		for triple in self.valid_set:
			h, r, t = triple
			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)   
			groundtruth['valid']['head'].setdefault((r, t), [])
			groundtruth['valid']['head'][(r, t)].append(h)
			groundtruth['valid']['tail'].setdefault((r, h), [])
			groundtruth['valid']['tail'][(r, h)].append(t) 

		for triple in self.test_set:
			h, r, t = triple


			groundtruth['all']['head'].setdefault((r, t), [])
			groundtruth['all']['head'][(r, t)].append(h)
			groundtruth['all']['tail'].setdefault((r, h), [])
			groundtruth['all']['tail'][(r, h)].append(t)
			groundtruth['all']['rel'].setdefault((h, t), [])
			groundtruth['all']['rel'][(h, t)].append(r)   
			groundtruth['test']['head'].setdefault((r, t), [])
			groundtruth['test']['head'][(r, t)].append(h)
			groundtruth['test']['tail'].setdefault((r, h), [])
			groundtruth['test']['tail'][(r, h)].append(t) 


		return groundtruth, possible_entities


	def get_groundtruth(self):
		return self.groundtruth

	def get_dataset(self, split):
		assert (split in ['train', 'valid', 'test'])
		
		if split == 'train':
			return self.train_set
		elif split == 'valid':
			return self.valid_set
		elif split == 'test':
			return self.test_set

	def my_tokenize(self, batch_tokens,batch_types, max_length=512, padding=True, model='roberta'):
		batch_size = len(batch_tokens)
		longest = min(max([len(i) for i in batch_tokens]), 512)

		if model == 'bert':
			input_ids = torch.zeros((batch_size, longest)).long()
		elif model == 'roberta':
			input_ids = torch.ones((batch_size, longest)).long()
		else:
			input_ids = torch.zeros((batch_size, longest)).long()

		token_type_ids = torch.zeros((batch_size, longest)).long()
		attention_mask = torch.zeros((batch_size, longest)).long()

		for i in range(batch_size):
			tokens = self.tokenizer.convert_tokens_to_ids(batch_tokens[i])
			input_ids[i, :len(tokens)] = torch.tensor(tokens).long() 
			attention_mask[i, :len(tokens)] = 1
			token_type_ids[i, :len(tokens)] =torch.tensor(np.array(batch_types[i])).long()

		if model == 'roberta':
			return BatchEncoding(data = {'input_ids': input_ids, 'attention_mask': attention_mask})
		elif model == 'bert':
			return BatchEncoding(data = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
		else:
			return BatchEncoding(
				data={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
