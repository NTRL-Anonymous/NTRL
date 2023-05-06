import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import pickle
import random
import copy
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self, lm_model, n_ent, n_rel, model ,score_function,hr,tr,ht,hrt):
		super().__init__()

		self.lm_model = lm_model
		self.n_ent = n_ent
		self.n_rel = n_rel
		self.model = model
		self.score_function = score_function
		self.hr = hr
		self.tr = tr
		self.ht = ht
		self.hrt = hrt
		self.num_cross = self.hr + self.tr + self.ht + hrt

		self.hidden_size = lm_model.config.hidden_size
		self.linear = nn.Linear(self.hidden_size,self.hidden_size)
		self.rel_embeddings = torch.nn.Embedding(n_rel, self.hidden_size)
		self.conv1 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=2)
		self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=2)
		if score_function == "mln":
			self.sim_classifier = nn.Sequential(nn.Linear(self.hidden_size * 3 , self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, 1))
		elif score_function == "cross_mln":
			self.sim_classifier = nn.Sequential(nn.Linear(self.hidden_size * (3+self.num_cross), self.hidden_size),
												nn.ReLU(),
												nn.Linear(self.hidden_size, 1))
		self.init_rel = False
		self.normalize_embs = False
		if 	score_function == 'transe':
			self.score_fn = self.score_triples_transe
			self.init_rel = True
			self.normalize_embs = True
		elif score_function == 'distmult':
			self.score_fn = self.score_triples_distmult
			self.init_rel = True
		elif score_function == 'complex':
			self.score_fn = self.score_triples_complex
			self.init_rel = True
		elif score_function == 'simple':
			self.score_fn = self.score_triples_simple
			self.init_rel = True
		elif score_function == 'mln':
			self.score_fn = self.score_triples_mln
		elif score_function == 'cross_mln':
			self.score_fn = self.score_triples_cross_mln
		else:
			raise ValueError(f'Unknown relational model {score_function}.')
		if self.init_rel:
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

	def forward(self, inputs):
		if self.model == 'NTRL'or self.model == 'BLP':
			batch_size = len(inputs['input_ids'])
			lm_model = self.lm_model
			device = lm_model.device
			embs = lm_model(**inputs).last_hidden_state[:,0,:]
			if self.normalize_embs:
				return F.normalize(self.linear(embs))
			else:
				return self.linear(embs)
		elif self.model == 'DKRL':
			lm_model = self.lm_model
			text_tok = inputs["input_ids"]

			text_mask = inputs["attention_mask"].to(torch.float)
			embs = lm_model.embeddings.word_embeddings(text_tok) * text_mask.unsqueeze(dim=-1)
			embs = embs.transpose(1, 2)
			text_mask = text_mask.unsqueeze(1)
			embs = F.pad(embs, [0, 1])
			embs = self.conv1(embs)
			embs = embs * text_mask
			if embs.shape[2] >= 4:
				kernel_size = 4
			elif embs.shape[2] == 1:
				kernel_size = 1
			else:
				kernel_size = 2
			embs = F.max_pool1d(embs, kernel_size=kernel_size)
			text_mask = F.max_pool1d(text_mask, kernel_size=kernel_size)
			embs = torch.tanh(embs)
			embs = F.pad(embs, [0, 1])
			embs = self.conv2(embs)
			lengths = torch.sum(text_mask, dim=-1)
			embs = torch.sum(embs * text_mask, dim=-1) / lengths
			embs = torch.tanh(embs)
			if self.normalize_embs:
				return F.normalize(embs)
			else:
				return embs


	def score_triples_transe(self, h_embs, r_embs, t_embs):
		scores = 39-torch.norm(h_embs + r_embs - t_embs, dim=-1, p=1).reshape(1,-1)
		return scores

	def score_triples_distmult(self, h_embs, r_embs, t_embs):
		scores = torch.sum(h_embs * r_embs * t_embs, dim=-1).reshape(1, -1)
		return scores

	def score_triples_complex(self, h_embs, r_embs, t_embs):
		heads_re, heads_im = torch.chunk(h_embs, chunks=2, dim=-1)
		tails_re, tails_im = torch.chunk(t_embs, chunks=2, dim=-1)
		rels_re, rels_im = torch.chunk(r_embs, chunks=2, dim=-1)

		scores =  torch.sum(rels_re * heads_re * tails_re +
						 rels_re * heads_im * tails_im +
						 rels_im * heads_re * tails_im -
						 rels_im * heads_im * tails_re,
						 dim=-1).reshape(1,-1)
		return scores
	def score_triples_simple(self, h_embs, r_embs, t_embs):
		heads_h, heads_t = torch.chunk(h_embs, chunks=2, dim=-1)
		tails_h, tails_t = torch.chunk(t_embs, chunks=2, dim=-1)
		rel_a, rel_b = torch.chunk(r_embs, chunks=2, dim=-1)

		scores = (torch.sum(heads_h * rel_a * tails_t +
						 tails_h * rel_b * heads_t, dim=-1) / 2).reshape(1,-1)
		return scores

	def score_triples_mln(self, h_embs, r_embs, t_embs):
		cat = torch.cat([h_embs,t_embs,r_embs ], dim=-1)
		scores = self.sim_classifier(cat).T
		return scores

	def score_triples_cross_mln(self, h_embs, r_embs, t_embs):
		if self.hr:
			hr_cross = h_embs * r_embs
		else:
			hr_cross = torch.tensor([]).to(self.lm_model.device)

		if self.ht:
			ht_cross = h_embs * t_embs
		else:
			ht_cross = torch.tensor([]).to(self.lm_model.device)

		if self.tr:
			tr_cross = t_embs * r_embs
		else:
			tr_cross = torch.tensor([]).to(self.lm_model.device)

		if self.hrt:
			hrt_cross = h_embs * r_embs * t_embs
		else:
			hrt_cross = torch.tensor([]).to(self.lm_model.device)

		cat = torch.cat([h_embs,t_embs,r_embs,hr_cross,tr_cross,ht_cross,hrt_cross], dim=-1)
		scores = self.sim_classifier(cat).T
		return scores

	def match(self, target_preds, target_encodeds,rel_embs):
		device = self.lm_model.device
		target_encoded_add = torch.cat([target_preds , target_encodeds] , axis = 0)
		sim_t = torch.zeros(target_preds.shape[0], target_encoded_add.shape[0]).to(self.lm_model.device)
		sim_h = torch.zeros(target_preds.shape[0], target_encoded_add.shape[0]).to(self.lm_model.device)

		for it, target_pred in enumerate(target_preds):
			rel_emb = rel_embs[it]
			rel_emb = rel_emb.expand(target_encoded_add.shape[0], target_pred.shape[0])
			target_pred = target_pred.expand(target_encoded_add.shape[0], target_pred.shape[0])
			score = self.score_fn(target_pred,rel_emb,target_encoded_add)
			sim_t[it] = score

		for it, target_encoded in enumerate(target_encodeds):
			rel_emb = rel_embs[it]
			rel_emb = rel_emb.expand(target_encoded_add.shape[0], target_encoded.shape[0])
			target_encoded = target_encoded.expand(target_encoded_add.shape[0], target_encoded.shape[0])
			score = self.score_fn(target_encoded_add,rel_emb,target_encoded)
			sim_h[it] = score
		return sim_t,sim_h
	def test_step(self , triples , ent_embs,ent2id,rel2id , mode):
		scores = []
		for tri in triples:
			h,r,t = tri
			h, r, t = ent2id[h],rel2id[r],ent2id[t]
			if mode == 'head-batch':
				score = []
				h_embs = ent_embs
				r_embs = self.rel_embeddings(torch.tensor(r).to(self.lm_model.device)).expand(h_embs.shape[0], h_embs.shape[1])
				t_embs = ent_embs[t].expand(h_embs.shape[0], h_embs.shape[1])
				score = self.score_fn(h_embs,r_embs,t_embs)
				b = torch.sigmoid(score)
				scores.append(b[0].tolist())
			else:
				score = []
				t_embs = ent_embs
				r_embs = self.rel_embeddings(torch.tensor(r).to(self.lm_model.device)).expand(t_embs.shape[0], t_embs.shape[1])
				h_embs = ent_embs[h].expand(t_embs.shape[0], t_embs.shape[1])
				score = self.score_fn(h_embs, r_embs, t_embs)
				b = torch.sigmoid(score)
				scores.append(b[0].tolist())
		return scores
