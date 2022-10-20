import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import sys
import json
from pathlib import Path

def read(path):
		""" Read the path to the json file searching for dataset from code """
		print("Reading Files...")
		path = Path(path)
		#with open(sys.argv[1]) as f:
		with open(path, 'rb') as f:
				squad_dict = json.load(f)

		contexts = []
		questions = []
		answers = []
		for group in squad_dict['data']:
				for passage in group['paragraphs']:
						context = passage['context']
						for qa in passage['qas']:
								question = qa['question']
								for answer in qa['answers']:
										contexts.append(context)
										questions.append(question)
										answers.append(answer)
		print("Read File : OK")
		return contexts, questions, answers

def read_from_terminal():
	""" Read the path to the json file searching for dataset from the terminal """
	print("Reading Files...")
	with open(sys.argv[1]) as f:
			squad_dict = json.load(f)

	contexts = []
	questions = []
	answers = []
	for group in squad_dict['data']:
			for passage in group['paragraphs']:
					context = passage['context']
					for qa in passage['qas']:
							question = qa['question']
							for answer in qa['answers']:
								contexts.append(context)
								questions.append(question)
								answers.append(answer)
	print("Read File : OK")
	return contexts, questions, answers

def processC(contexts):
	""" Process the context list to a conform datafram """
	Processed_contexts = pd.DataFrame(list(dict.fromkeys(contexts)))
	Processed_contexts.columns = ['Contexts']

	return Processed_contexts

def processQ(question):
	""" Process the question list to a conform datafram """
	Processed_questions = pd.DataFrame(list(dict.fromkeys(question)))
	Processed_questions.columns = ['Questions']
	return Processed_questions

def tfifd_components(context):
	""" Computes the TF-IDF matrix from the contexts."""
	print("Start TF-IDF")
	vectorizer = TfidfVectorizer()
	print("TF-IDF matrix computed")
	TF_IDF_matrix = vectorizer.fit_transform(context)

	print('Vocabulary Size : ', len(vectorizer.get_feature_names()))
	print('Shape of Matrix : ', TF_IDF_matrix.shape)
	return vectorizer,TF_IDF_matrix

def read_nb_contexts():
	var = input("Please enter the number of contexts needed: ")
	return var
	
def calculate_similarity(X, vectorizor, question,top_k ):
	""" Vectorizes the question via `vectorizor` and calculates the cosine similarity of
	the question and the contexts and returns the `top_k` similar contexts."""
	print("Computing Similarity ...")
	question_vec = vectorizor.transform(question)
	cosine_similarities = cosine_similarity(X,question_vec).flatten()
	most_similar_context_indices = np.argsort(cosine_similarities, axis=0)[:-top_k-1:-1]
	return (most_similar_context_indices, cosine_similarities)

def retrieve_context(ProbContIndices,contextsFrame,k):
	""" Sort the top retrieved contexts """
	retrieved_articles = []
	for rank, sort_index in enumerate(ProbContIndices):
		retrieved_articles.append(contextsFrame[sort_index])
		#print ('Rank : ', rank,' context : ', contextsFrame[sort_index])
			
	return retrieved_articles[:k]

def show_retrieve_context(ProbContIndices,contextsFrame,k):
	""" Print the retrieved contexts """
	for rank, sort_index in enumerate(ProbContIndices):
		print ('\nRank : ', rank,' context : ', contextsFrame[sort_index])


def read_question ():
	""" Get the question from the terminal """
	var = input("Please enter a query: ")
	return var


def mean_rank(nb_questions,joined_frames,Tf,V,context):
	""" Calculate the mean rank for the correct context for the full dataset """
	rank_sum = 0
	print("Computing the mean_rank ...")
	for i in range(nb_questions) :
		query = joined_frames["Questions"][i]
		query = [query.replace('?','')]
		most_similar_context_indices, cosine_similarities = calculate_similarity(Tf,V,query,nb_questions)
		retrieved_article = retrieve_context(most_similar_context_indices,context['Contexts'],nb_questions)
		true_article = joined_frames["Contexts"][i]
		for i in range(nb_questions):
			if true_article == retrieved_article[i] :
				rank_sum += i
				break
	return rank_sum/nb_questions

def top_1_accuracy (nb_questions,joined_frames,Tf,V,context):
	""" Calculate the top_1 accuracy accross the full dataset and return the mean retrieval time as Indicator """
	print("Computing the top_1 accuracy ...")
	nb_nice = 0
	retrieval_time = 0
	for i in range(nb_questions) :
		query = joined_frames["Questions"][i]
		query = [query.replace('?','')]
		start_time = time.time()
		most_similar_context_indices, cosine_similarities = calculate_similarity(Tf,V,query,4)
		retrieved_article = retrieve_context(most_similar_context_indices,context['Contexts'],1)
		search_time = time.time()-start_time
		retrieval_time += search_time
		true_article = joined_frames["Contexts"][i]
		if true_article == retrieved_article[0] : 
			nb_nice += 1
	return nb_nice/nb_questions,retrieval_time/nb_questions

def top_4_accuracy (nb_questions,joined_frames,Tf,V,context):
	""" Calculate the top_4 accuracy accross the full dataset """
	nb_nice = 0
	print("Computing the top_4 accuracy ...")
	for i in range(nb_questions) :
		query = joined_frames["Questions"][i]
		query = [query.replace('?','')]
		most_similar_context_indices, cosine_similarities = calculate_similarity(Tf,V,query,4)
		retrieved_article = retrieve_context(most_similar_context_indices,context['Contexts'],4)
		true_article = joined_frames["Contexts"][i]
		if (true_article == retrieved_article[0] or true_article == retrieved_article[1] or true_article == retrieved_article[2] or true_article == retrieved_article[3]) : 
			nb_nice += 1
	return nb_nice/nb_questions

def join_dataframes(context,questions):
	""" Create a pair of question and context dataframe """
	return pd.DataFrame(questions,columns=["Questions"]).join(pd.DataFrame(context,columns=["Contexts"]))
