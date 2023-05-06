import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import time
import math
import os
import pickle
import numpy as np
import logging
from datetime import datetime

class Trainer:
	def __init__(self, data_loader, model, tokenizer, optimizer, device, hyperparams):

		self.data_loader = data_loader
		self.model = model

		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.device = device
		self.hyperparams = hyperparams
		self.data = hyperparams['data']
		self.lamda = hyperparams['lamda']

		model.to(device)

	def run(self):
		self.train()

	def train(self):
		model = self.model
		tokenizer = self.tokenizer
		optimizer = self.optimizer

		device = self.device
		hyperparams = self.hyperparams

		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch']

		data_loader = self.data_loader
		ent2id = data_loader.ent2id
		rel2id = data_loader.rel2id
		entity_list = data_loader.entity_list
		relation_list = data_loader.relation_list
		groundtruth = data_loader.get_groundtruth()

		# criterion 
		criterion = torch.nn.CrossEntropyLoss()
		bce_criterion = torch.nn.BCELoss(reduction='none')
		sigmoid = torch.nn.Sigmoid()

		model.train()
		best_epoch = 0
		best_metrics_valid = {
			'MRR': 0.0,
			'MR': 0.0,
			'HITS@1': 0.0,
			'HITS@3': 0.0,
			'HITS@10': 0.0,
		}
		best_metrics_test = {
			'MRR': 0.0,
			'MR': 0.0,
			'HITS@1': 0.0,
			'HITS@3': 0.0,
			'HITS@10': 0.0,
		}
		for epc in range(epoch):
			total_loss_link_prediction = 0
			time_begin = time.time()
			data_sampler = data_loader.train_data_sampler()
			n_batch = len(data_sampler)
			dataset_size = data_sampler.get_dataset_size()
			real_dataset_size = dataset_size
			for i_b, batch in tqdm(enumerate(data_sampler), total=n_batch,desc = 'train'):
				triples = [i[0] for i in batch]
				batch_size_ = len(batch)
				real_idxs = [ _ for _, i in enumerate(batch) if i[1] == 1]
				real_triples = [ i[0] for _, i in enumerate(batch) if i[1] == 1]
				real_batch_size = len(real_triples)
				real_inputs_h,real_inputs_t = data_loader.batch_tokenize(real_triples)
				real_inputs_h.to(device)
				real_inputs_t.to(device)
				label_idx_list = []
				labels_t = torch.zeros((len(real_triples), 2 * len(real_triples))).to(device)
				targets = [ triple[0] for triple in real_triples] + [ triple[2] for triple in real_triples]
				target_idxs = [ ent2id[tar] for tar in targets]
				for i, triple in enumerate(real_triples):
					h, r, t = triple
					expects = set(groundtruth['train']['tail'][(r, h)])
					label_idx = [i_t for i_t, target in enumerate(targets) if target in expects]
					label_idx_list.append(label_idx)
					labels_t[i, label_idx] = 1
				labels_h = torch.zeros((len(real_triples), 2 * len(real_triples))).to(device)

				targets = [triple[0] for triple in real_triples] + [triple[2] for triple in
																	real_triples]
				target_idxs = [ent2id[tar] for tar in targets]
				for i, triple in enumerate(real_triples):
					h, r, t = triple
					expects = set(groundtruth['train']['head'][(r, t)])
					label_idx = [i_t for i_t, target in enumerate(targets) if target in expects]
					label_idx_list.append(label_idx)
					labels_h[i, label_idx] = 1
				labels = torch.cat([labels_h , labels_t])
				loss = 0

				bce_loss = []
				preds_list = []


				target_preds = model(real_inputs_h)
				target_encodes = model(real_inputs_t)
				rel_embs = model.rel_embeddings(torch.LongTensor([rel2id[triple[1]] for triple in real_triples]).to(device))
				preds_t ,preds_h  = model.match( target_preds, target_encodes,rel_embs)
				preds = torch.cat([preds_h , preds_t] , axis =0)
				preds = sigmoid(preds)
				bce_loss.append(bce_criterion(preds, labels))
				preds_list.append(preds)
				pred_labels = (preds > 0.5).int() # 二分类
				label_idx_list_new = []
				label_idx_list_new.extend(label_idx_list[len(label_idx_list) // 2:])
				label_idx_list_new.extend(label_idx_list[:len(label_idx_list) // 2])
				label_idx_list = label_idx_list_new
				for i in range(2 * real_batch_size):

					pos_idx = sorted(label_idx_list[i])
					pos_set = set(pos_idx)
					neg_idx = [ _ for _ in range(labels.shape[1]) if not _ in pos_set]

					for j, bl in enumerate(bce_loss):
						# separately add lm_loss, transe_loss, and ensembled_loss
						l = bl[i]
						pos_loss = l[pos_idx].mean()
						a = 1
						if hyperparams['self_adversarial']:
							# self-adversarial sampling
							neg_selfadv_weight = preds_list[j][i][neg_idx]  # selfadv_weight[i][neg_idx]
							neg_weights = neg_selfadv_weight.softmax(dim=-1)
							neg_loss = (l[neg_idx] * neg_weights).sum()
							a = torch.max(neg_weights)
						else:
							neg_loss = l[neg_idx].mean()
						if self.lamda == 'lamda1':
							loss +=  a*pos_loss + neg_loss
						else:
							loss += pos_loss + neg_loss


				total_loss_link_prediction += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			avg_loss_link_prediction = total_loss_link_prediction / real_dataset_size

			time_end = time.time()
			time_epoch = time_end - time_begin
			logging.info('Train: Epoch: {} , Avg_Link_Prediction_Loss: {}, Time: {}'.format(
				epc, avg_loss_link_prediction, time_epoch))

			metrics_valid = self.link_prediction(epc,'valid')
			metrics_test = self.link_prediction(epc, 'test')
			best_valid_mrr = best_metrics_valid["MRR"]
			valid_mrr = metrics_valid["MRR"]
			if valid_mrr > best_valid_mrr:
				best_metrics_valid = metrics_valid
				best_metrics_test = metrics_test
				best_epoch = epc
		logging.info('Best_Epoch: {}'.format(
				best_epoch))
		logging.info('Best_Test: {}'.format(
			best_metrics_test))






	def link_prediction(self, epc=-1, split='valid'):
		model = self.model
		device = self.device
		hyperparams = self.hyperparams
		data_loader = self.data_loader

		n_ent = model.n_ent
		n_rel = model.n_rel

		ent2id = data_loader.ent2id
		rel2id = data_loader.rel2id
		entity_list = data_loader.entity_list

		model.eval()

		sigmoid = torch.nn.Sigmoid()
		dataset = data_loader.get_dataset(split)
		groundtruth = data_loader.get_groundtruth()
		ks = [1, 3, 10]
		MR = { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } 

		MRR = { 
				setting:
					{target: 0 for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 }

		hits = { 
				setting:
					{target: {k: 0 for k in ks} for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			 } 
		# calc target embeddings
		batch_size = 128
		ent_target_encoded = torch.zeros((n_ent, model.hidden_size)).to(device)
		rel_target_encoded = torch.zeros((n_rel, model.hidden_size)).to(device)
		logs = []
		with torch.no_grad():
			# calc entity target embeddings
			random_map = [ i for i in range(n_ent)]
			batch_list = [ random_map[i:i+batch_size] for i in range(0, n_ent, batch_size)]

			for batch in batch_list:
				batch_targets = [ entity_list[_] for _ in batch]
				target_inputs = data_loader.batch_tokenize_target(targets=batch_targets)
				target_inputs.to(device)
				target_encodes = model(target_inputs)
				ent_target_encoded[batch] = target_encodes
			for mode in ['tail-batch','head-batch']:
				start = 0
				batch_size_test = 1

				while start < len(dataset):
					if start + batch_size_test < len(dataset):
						end = start + batch_size_test
					else:
						end = len(dataset)

					scores = model.test_step(dataset[start:end],ent_target_encoded,ent2id,rel2id,mode)

					for i, t in enumerate(dataset[start:end]):
						head, relation, tail = t
						if mode == 'head-batch':
							trues = set(groundtruth['all']['head'][relation, tail]) - {head}
						else:
							trues = set(groundtruth['all']['tail'][relation, head]) - {tail}
						for ts in list(trues):
							scores[i][ent2id[ts]]-=100
					scores = torch.tensor(np.array(scores)).to(device)
					argsort_heads = torch.argsort(scores, dim=1, descending=True)
					if mode == 'head-batch':

						positive_arg_heads = torch.LongTensor([ent2id[i[0]] for i in dataset[start:end]]).to(device)
					else:
						positive_arg_heads = torch.LongTensor([ent2id[i[2]] for i in dataset[start:end]]).to(device)
					for i in range(len(dataset[start:end])):
						# Notice that argsort is not ranking
						ranking = (argsort_heads[i, :] == positive_arg_heads[i]).nonzero()
						assert ranking.size(0) == 1

						# ranking + 1 is the true ranking used in evaluation metrics
						ranking = 1 + ranking.item()
						logs.append({
							'MRR': 1.0 / ranking,
							'MR': float(ranking),
							'HITS@1': 1.0 if ranking <= 1 else 0.0,
							'HITS@3': 1.0 if ranking <= 3 else 0.0,
							'HITS@10': 1.0 if ranking <= 10 else 0.0,
						})
					start = start + batch_size_test

			metrics = {}
			for metric in logs[0].keys():
				metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
			logging.info(metrics)
			return metrics