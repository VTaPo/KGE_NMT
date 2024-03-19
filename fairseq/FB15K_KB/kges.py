from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.datasets import FB15k
from transformers import pipeline
import requests
import string
from collections import defaultdict

from fairseq.FB15K_KB.semantic_sim import get_maxIndex_semanticSim

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

def create_Qid_MID_dict(mapping_file_path):
	result_dict = {}
	with open(mapping_file_path, 'r', encoding='utf-8') as file:
		for line in file:
			key, value = line.strip().split('\t')
			result_dict[key] = value

	return result_dict

def create_realName_Qid_dict(mapping_file_path):
	result_dict = defaultdict(list)
	with open(mapping_file_path, 'r', encoding='utf-8') as file:
		for line in file:
			key, value = line.strip().split('\t')
			result_dict[key].append(value)

	return result_dict

def create_Qid_description_dict(mapping_file_path):
	result_dict = {}
	with open(mapping_file_path, 'r', encoding='utf-8') as file:
		for line in file:
			key, value = line.strip().split('\t')
			result_dict[key] = value

	return result_dict

def mapping_Qid_MID(Qid, Qid_MID_dict):
	return Qid_MID_dict.get(Qid,'unk')

def mapping_realName_Qid(realName, realName_Qid_dict):
	return realName_Qid_dict.get(realName,['unk'])

def mapping_Qid_description(Qid, Qid_description_dict):
	return Qid_description_dict.get(Qid,'unk')

def remove_non_alphabetic(sentence):
	valid_characters = set(string.ascii_lowercase) | set(string.ascii_uppercase) | {' '}
	cleaned_sentence = ''.join(char for char in sentence if char in valid_characters)
	return cleaned_sentence

def get_KGE(model, dataset, ner_model, sem_model, text, name_Q_dict, Q_M_dict, Q_desc_dict, dim=512):
	row = [torch.zeros(dim).to('cuda')]
	sentence = ' '.join(text)
	sentence = remove_non_alphabetic(sentence)
	ner_results = ner_model(sentence)
	ents = [result['word'] for result in ner_results]
	if len(ents) == 0:
		return torch.stack(row), False
	qids = list(map(lambda ent: mapping_realName_Qid(ent, name_Q_dict), ents))
	for i in range(len(qids)):
		if len(qids[i]) == 1:
			qids[i] = qids[i][0]
		else:
			list_text = list(map(lambda qid: mapping_Qid_description(qid, Q_desc_dict), qids[i]))
			max_idx = get_maxIndex_semanticSim(sentence, list_text, sem_model)
			qids[i] = qids[i][max_idx]
	mids = list(map(lambda qid: mapping_Qid_MID(qid, Q_M_dict), qids))
	filtered_mids = list(filter(lambda x: x != 'unk', mids))
	if len(filtered_mids) == 0:
		return torch.stack(row), False
	list_idx = dataset.training.entities_to_ids(filtered_mids)
	_idx = torch.as_tensor(list_idx, device=model.device)
	entity_embeddings = model.entity_representations[0]
	_embedding = list(entity_embeddings(_idx).detach())
	return torch.stack(_embedding), True
	

def padding_KGE(max_dim, embeddings):
	padding_size = max_dim - embeddings.shape[0]
	if padding_size!=0:
		padding = torch.zeros((padding_size, embeddings.shape[1]), dtype=embeddings.dtype).to('cuda')
		embeddings = torch.cat((embeddings, padding), dim=0)
		return embeddings
	else:
		return embeddings

def get_final_KGE_batch(model, dataset, ner_model, sem_model, texts_list, name_Q_dict, Q_M_dict, Q_desc_dict, dim=512):
	batch_embeddings = []
	flags = []

	results = map(lambda text: get_KGE(model, dataset, ner_model, sem_model, text, name_Q_dict, Q_M_dict, Q_desc_dict), texts_list)
	batch_embeddings, flags = map(list,zip(*results))

	max_dim_0 = max(tensor.shape[0] for tensor in batch_embeddings)
	batch_embeddings = list(map(lambda embedds: padding_KGE(max_dim_0, embedds), batch_embeddings))
	batch_embeddings = torch.stack(batch_embeddings)
	return batch_embeddings, flags

def KGEs(model, texts, ner, sem, dataset, name_Q_dict, Q_M_dict, Q_desc_dict):
	KGEs, flags = get_final_KGE_batch(model, dataset, ner, sem, texts, name_Q_dict, Q_M_dict, Q_desc_dict)

	return KGEs, flags

