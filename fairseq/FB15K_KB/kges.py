from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.datasets import FB15k
from transformers import pipeline
import requests

import numpy as np

import torch

def get_qid(entity_name):
	url = f"https://www.wikidata.org/w/api.php"
	params = {
		"action": "wbsearchentities",
		"search": entity_name,
		"language": "en",
		"format": "json"
	}

	response = requests.get(url, params=params)
	data = response.json()

	if "search" in data and data["search"]:
		# Lấy Qid của kết quả đầu tiên (nếu có)
		qid = data["search"][0]["id"]
		return qid
	else:
		return '-Q'

def create_Qid_MID_dict(mapping_file_path, entities_file_path):
	mids_FB15k = []
	with open(entities_file_path, 'r', encoding='utf-8') as f:
		for line in f:
			k, v = line.strip().split('\t')
			mids_FB15k.append(v)

	result_dict = {}
	with open(mapping_file_path, 'r', encoding='utf-8') as file:
		for line in file:
			key, value = line.strip().split('\t')
			if key in mids:
				result_dict[value] = key

	return result_dict

def mapping_Qid_MID(Qid, Qid_MID_dict):
	return Qid_MID_dict.get(Qid,'unk')

def get_KGE(model, dataset, ner_model, text, Q_M_dict, dim=512):
	row=[]
	for i in range(len(text)):
		row.append(torch.zeros(dim).to('cuda'))
	mark={}
	idx = [0 for _ in range(len(text))]
	sentence = ' '.join(text)
	ner_results = ner_model(sentence)
	ents = [result['word'] for result in ner_results]
	for e in ents:
		for i in range(len(text)):
			if e == text[i]:
				idx[i]=1
	for i in range(len(idx)):
		if idx[i]!=0:
			qid = get_qid(text[i])
			if qid == '-Q':
				pass
			else:
				m = mapping_Qid_MID(qid, Q_M_dict)
				if m == 'unk':
					pass
				else:
					list_idx = dataset.training.entities_to_ids([m])
					_idx = torch.as_tensor(list_idx, device=model.device)
					entity_embeddings = model.entity_representations[0]
					_embedding = entity_embeddings(_idx).detach()
					row[i]=_embedding[0]
	row = torch.stack(row)
	return row
	# Qids = []
	# MIDs = []

	# if len(ents)==0:
	# 	return torch.stack(row)
	# else:
	# 	for ent in ents:
	# 		qid = get_qid(ent)
	# 		if qid=='-Q':
	# 			for i in range(len(text)):
	# 				if ent == text[i]:
	# 					idx[i]=0
	# 		else:
	# 			Qids.append(qid)
	# 			temp = []
	# 			for i in range(len(text)):
	# 				if ent == text[i]:
	# 					temp.append(i)
	# 			mark[qid]=temp
	# 	if len(set(Qids))==1:
	# 		return torch.stack(row)
	# 	else:
	# 		for q in Qids:
	# 			m = mapping_Qid_MID(q, Q_M_dict)
	# 			if m == 'unk':
	# 				for i in mark[q]:
	# 					idx[i]=0
	# 				mark[q]=-1
	# 			else:
	# 				MIDs.append(m)
	# 		if len(MIDs)==0:
	# 			return torch.stack(row)
	# 		mark_mid = []
	# 		for k in list(mark.keys()):
	# 			if mark[k]!=-1:
	# 				for _idx in mark[k]:
	# 					mark_mid.append(_idx)
	# 		mark_mid.sort()
	# 		list_idx = dataset.training.entities_to_ids(MIDs)
	# 		idx = torch.as_tensor(list_idx, device=model.device)
	# 		entity_embeddings = model.entity_representations[0]
	# 		_embedding = entity_embeddings(idx).detach()
	# 		for i in range(len(mark_mid)):
	# 			row[mark_mid[i]]=_embedding[i]
	# 		row = torch.stack(row)
	# 		return row

def get_final_KGE_batch(model, dataset, ner_model, texts_list, Q_M_dict, dim=512):
	batch_embeddings = []

	for text in texts_list:
		embeddings = get_KGE(model, dataset, ner_model, text, Q_M_dict)
		batch_embeddings.append(embeddings)
	batch_embeddings = torch.stack(batch_embeddings)

	return batch_embeddings

def KGEs(texts):
	model = torch.load('fairseq/FB15K_KB/fb15k_transe/trained_model.pkl')

	dataset = FB15k()

	ner = pipeline("ner", grouped_entities=True)

	fb2w_file_path = 'fairseq/FB15K_KB/fb2w.txt'
	Q_M_dict = create_Qid_MID_dict(fb2w_file_path)

	KGEs = get_final_KGE_batch(model, dataset, ner, texts, Q_M_dict)

	return KGEs

