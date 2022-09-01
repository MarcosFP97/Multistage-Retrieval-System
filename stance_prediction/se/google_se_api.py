import re
import nltk
import pandas as pd
import argparse
from http.client import parse_headers
from statistics import mean
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import numpy as np
from langdetect import detect
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from sentence_transformers import SentenceTransformer, util
import torch
torch.cuda.set_device(1)
print(torch.cuda.current_device()) # checks if GPU is available
nltk.download('punkt')

class GoogleCSE:
	def __init__(self):
		# get the API KEY here: https://developers.google.com/custom-search/v1/overview
		self.API_KEY = "AIzaSyCDXsV-uNZjMHH2Fm9RLTlY4XzJJrClzWw"
		# get your Search Engine ID on your CSE control panel
		self.SEARCH_ENGINE_ID = "6de49215ca8b05e76"
		page = 1
		self.start = (page - 1) * 10 + 1

	def search(self, query):
		print("Query:", query)
		with open("google_search_2022.txt", "a") as f:
			f.write("Query: " + query + '\n')
		url = f"https://www.googleapis.com/customsearch/v1?key={self.API_KEY}&cx={self.SEARCH_ENGINE_ID}&q={query}&start={self.start}"
		# make the API request
		data = requests.get(url).json()

		# get the result items
		search_items = data.get("items")
		# iterate over 10 results found
		sites = []
		for i, search_item in enumerate(search_items, start=1):
			# extract the page url
			link = search_item.get("link")
			sites.append(link)
			# print the results
			print("="*10, f"Result #{i+self.start-1}", "="*10+ '\n')
			print("URL: ", link, "\n")
			with open("google_search_2022.txt", "a") as f:
				f.write("="*10 + f"Result #{i+self.start-1}" + "="*10 +'\n')
				f.write("URL: "+link + '\n')

		with open("google_search.txt", "a") as f:
				f.write("*"*10+'\n')
				f.write('\n')
		print("*"*10+'\n')
		print()
		return sites

class StancePredictor:
	def __init__(self, url, query, number):
		self.url = url
		self.query = query
		df = pd.read_csv('trec-pipeline/stance_prediction/se/topic_variants_2022.txt', names=["query", "affirmative", "negative"])
		self.claim_helpful = df[df["query"]==query]["affirmative"].values[0] 
		self.claim_unhelpful = df[df["query"]==query]["negative"].values[0] 
		print(self.url)
		print(self.claim_helpful)
		print(self.claim_unhelpful)
		print("Rank", number)
	
	'''
	This function creates a sliding window over text
	'''
	def window(self, iterable, n = 6, m = 3):
		if m == 0: # otherwise infinte loop
			raise ValueError("Parameter 'm' can't be 0")
		lst = list(iterable)
		i = 0

		if i + n < len(lst):
			while i + n < len(lst):
				yield ' '.join(lst[i:i + n])
				i += m

		else:
			yield ' '.join(lst)

	def scrape_web(self):
		try:
			r = requests.get(self.url)
			soup = BeautifulSoup(r.content, "html.parser")

			for script in soup(["script", "style"]): # tag removal
				script.decompose() 
				
			doc = soup.get_text()
			r.raise_for_status()
		except requests.exceptions.RequestException as e:
			doc = ""

		return doc
	
	def passage_extract(self, doc, reranker):

		try:
			lang = detect(doc)
		except:
			lang = "error"

		passage = ""
		score = ""
		if lang == "en":
			doc = nltk.sent_tokenize(doc)
			doc_passages = self.window(doc, 6, 3) # len = 6, stride = 3
			texts = [Text(p.encode("utf-8", errors="replace").decode("utf-8"), None, 0) for p in doc_passages]
			
			query = Query(self.query)
			reranked = reranker.rerank(query, texts)
			reranked.sort(key=lambda x: x.score, reverse=True)
			if reranked:
				score = reranked[0].score
				passage = reranked[0].text # we select the top 1 reranked passage and score (= most relevant passage for the query)
		return passage, score

	def comp_sim(self, passage, embedder):
		
		l_passages = nltk.sent_tokenize(passage)
		help_embedding = embedder.encode(self.claim_helpful, convert_to_tensor=True)
		unhelp_embedding = embedder.encode(self.claim_unhelpful, convert_to_tensor=True)
		
		sims_help = []
		sims_unhelp = []
		for sent in l_passages:
			sent_embeddings = embedder.encode(sent, convert_to_tensor=True)
			sim_help = util.pytorch_cos_sim(help_embedding, sent_embeddings)[0]
			sims_help.append(float(sim_help))
			sim_unhelp = util.pytorch_cos_sim(unhelp_embedding, sent_embeddings)[0]
			sims_unhelp.append(float(sim_unhelp))

		return mean(sims_help), mean(sims_unhelp)

if __name__ == "__main__":
	# gcse = GoogleCSE()
	# root = ET.parse('../evaluation/misinfo-2022-topics.xml').getroot()
	# for topic in root.findall('topic'):
	# 	description = topic.find("question").text
	# 	sites = gcse.search(description)

	reranker =  MonoT5('castorini/monot5-base-med-msmarco')
	embedder =  SentenceTransformer('stsb-roberta-base-v2')

	with open('trec-pipeline/stance_prediction/se/google_search_2022_1.txt', 'r') as f:
		lines = f.readlines()
		query = ""
		url = ""
		number = ""
		spp_help = ""
		spp_unhelp = ""
		score = ""
		preds = []
		for line in lines:
			if line.startswith("Query:"):
				query = line.replace("Query:","").strip()
			elif line.startswith("URL:"):
				url = line.replace("URL:","").strip()
				if query and url and number:
					sp = StancePredictor(url, query, number)
					doc = sp.scrape_web()
					
					if doc:
						passage, score = sp.passage_extract(doc, reranker)
						if passage:
							spp_help, spp_unhelp = sp.comp_sim(passage, embedder)
							print("Helpful prediction!!!", spp_help)
							print("Unhelpful prediction!!!", spp_unhelp)
						if score:
							preds.append((query, number, url, spp_help, spp_unhelp, np.exp(score), passage))
						else:
							preds.append((query, number, url, spp_help, spp_unhelp, score, passage))
					url = ""
					number = ""
					spp_help = ""
					spp_unhelp = ""
			elif "*" in line:
				query = ""
			else:
				print(line)
				if "Result" in line:
					number = int(re.search("\d+", line)[0])
				continue
		
	df = pd.DataFrame(preds, columns=["query", "rank", "url", "pred_helpf", "pred_unhelpf", "score_passage", "passage"])
	print(df.head())
	df.to_csv('experiments/trec-pipeline/runs/trec-2022/stance_prediction_2022.csv', header=True, index=False)
			
		


