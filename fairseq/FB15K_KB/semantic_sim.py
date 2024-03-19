from sentence_transformers import util

def semanticSim(origin_text,compare_text,model):
	embedding_1= model.encode(origin_text, convert_to_tensor=True)
	embedding_2 = model.encode(compare_text, convert_to_tensor=True)
	return util.pytorch_cos_sim(embedding_1, embedding_2)

def get_maxIndex_semanticSim(origin_text,list_text,model):
	scores = list(map(lambda text: semanticSim(origin_text, text, model), list_text))
	index = scores.index(max(scores))
	return index