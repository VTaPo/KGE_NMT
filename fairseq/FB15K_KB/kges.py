from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.datasets import FB15k
from transformers import pipeline
import requests
import string

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
			result_dict[value] = key

	return result_dict

def mapping_Qid_MID(Qid, Qid_MID_dict):
	return Qid_MID_dict.get(Qid,'unk')

def remove_non_alphabetic(sentence):
	valid_characters = set(string.ascii_lowercase) | set(string.ascii_uppercase) | {' '}
	cleaned_sentence = ''.join(char for char in sentence if char in valid_characters)
	return cleaned_sentence

def get_KGE(model, dataset, ner_model, text, Q_M_dict, dim=512):
	row = [torch.zeros(dim).to('cuda')]
	sentence = ' '.join(text)
	sentence = remove_non_alphabetic(sentence)
	words = sentence.split()
	ner_results = ner_model(sentence)
	ents = [result['word'] for result in ner_results]
	if len(ents) == 0:
		return torch.stack(row), False
	qids = list(map(get_qid,ents))
	mids = list(map(lambda qid: mapping_Qid_MID(qid, Q_M_dict), qids))
	filtered_mids = list(filter(lambda x: x != 'unk', mids))
	if len(filtered_mids) == 0:
		return torch.stack(row), False
	list_idx = dataset.training.entities_to_ids(filtered_mids)
	_idx = torch.as_tensor(list_idx, device=model.device)
	entity_embeddings = model.entity_representations[0]
	_embedding = list(entity_embeddings(_idx).detach())
	return torch.stack(_embedding), True
	

def get_final_KGE_batch(model, dataset, ner_model, texts_list, Q_M_dict, dim=512):
	batch_embeddings = []
	flags = []

	for text in texts_list:
		embeddings, flag = get_KGE(model, dataset, ner_model, text, Q_M_dict)
		batch_embeddings.append(embeddings)
		flags.append(flag)

	max_dim_0 = max(tensor.shape[0] for tensor in batch_embeddings)
	for i in range(len(batch_embeddings)):
		padding_size = max_dim_0 - batch_embeddings[i].shape[0]
		if padding_size!=0:
		    padding = torch.zeros((padding_size, batch_embeddings[i].shape[1]), dtype=batch_embeddings[i].dtype).to('cuda')
		    batch_embeddings[i] = torch.cat((batch_embeddings[i], padding), dim=0)
	batch_embeddings = torch.stack(batch_embeddings)
	return batch_embeddings, flags

def KGEs(model, texts, ner, dataset, Q_M_dict):
	KGEs, flags = get_final_KGE_batch(model, dataset, ner, texts, Q_M_dict)

	return KGEs, flags

