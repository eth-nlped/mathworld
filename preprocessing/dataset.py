import jsonlines, json
import xml.etree.ElementTree as ET

import pandas as pd
import os
import random
import re
import spacy
from tqdm import tqdm
from fractions import Fraction

from preprocessing.split_clauses import *

class Dataset:

	def __init__(self, dataset, fold = None, split_sentences = False, split_questions = False, seed=None):
		random.seed(seed)
		if dataset == "gsm8k":
			self.load_gsm8k()
			self.remove_duplicates()
			if split_questions:
				self.split_questions()
				if split_sentences:
					self.split_sentences()
			self.name = "gsm8k"

		elif dataset == "mathqa":
			self.load_mathqa()
			self.name = "mathqa"

		elif dataset == "asdiv-a":
			self.load_asdiv(fold)
			self.remove_duplicates()
			if split_sentences:
				self.split_sentences()
			self.name = "asdiv-a"

		elif dataset == "svamp":
			self.load_svamp(fold)
			self.remove_duplicates()
			if split_sentences:
				self.split_sentences()
			self.name = "svamp"

		elif dataset == "mawps":
			self.load_mawps(fold)
			self.remove_duplicates()
			if split_sentences:
				self.split_sentences()
			self.name = "mawps"

		elif dataset == "all-arith":
			self.load_allarith()
			self.remove_duplicates()
			if split_questions:
				self.split_questions()
				if split_sentences:
					self.split_sentences()
			self.name = "all-arith"


	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def __str__(self):
		return self.name

	def load_gsm8k(self):
		with jsonlines.open("../data/gsm8k/train.jsonl") as f:
			data = []
			for obj in f:
				data.append(obj)
			newdata = []
			for i, elem in enumerate(data):
				if len(elem["question"]) < 5:
					print("Empty string")
					continue
				elem["id"] = "gsm8k-" + str(i)
				# remove extra whitespaces
				elem["problem"] = re.sub(' +', ' ', elem["question"]).replace(u'\xa0', u' ').strip()
				elem["answer"] = elem["answer"].split("\n#### ")[1].strip()
				newdata.append(elem)

		self.data = newdata

	def load_mathqa(self):
		with open("../data/MathQA/train.json") as f:
			self.data = json.load(f)

	def load_asdiv(self, fold):
		folds = sorted(os.listdir("../data/asdiv-a/folds"))
		problem_ids = []
		if fold == None:
			for subset in folds:
				with open(os.path.join("../data/asdiv-a/folds", subset)) as f:
					data = f.readlines()
					data = [id.replace('\n', '') for id in data]
					problem_ids.extend(data)
		else:
			subset = folds[fold]
			with open(os.path.join("../data/asdiv-a/folds", subset)) as f:
				data = f.readlines()
				data = [id.replace('\n', '') for id in data]
				problem_ids.extend(data)

		tree = ET.parse("../data/asdiv-a/ASDiv.xml")
		root = tree.getroot()
		problem_tags = root.findall("ProblemSet/Problem")

		data = []
		for tag in problem_tags:
			id = tag.attrib["ID"]
			if id not in problem_ids:
				continue
			body = tag.find("Body").text
			question = tag.find("Question").text
			answer = tag.find("Answer").text
			answer = re.findall("[0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+", answer)[0]
			answer = self.format_number(answer)
			data.append({"id": id,
						 "body": re.sub(' +', ' ', body).strip(),
						 "question": re.sub(' +', ' ', question).strip(),
						 "answer": str(answer).strip()})

		newdata = []
		for elem in data:
			newid = elem["id"].strip("nluds-")
			elem["id"] = "asdiv-" + newid
			newdata.append(elem)

		self.data = newdata

	def load_svamp(self, fold):
		df0, df1, df2, df3, df4 = map(pd.read_csv,
									  map("../data/svamp/cv_svamp_augmented/fold{}/dev.csv".format, [0, 1, 2, 3, 4]))
		dfs = [df0, df1, df2, df3, df4]
		if fold == None:
			df = pd.concat(dfs)
		else:
			df = dfs[fold]
		data = []
		for id, (_, row) in enumerate(df.iterrows()):
			body, question = self.impute_numbers(row["Body"], row["Ques"], row["Numbers"])
			answer = row["Answer"]
			answer = self.format_number(answer)
			data.append({"id": id,
						 "body": re.sub(' +', ' ', body).strip(),
						 "question": re.sub(' +', ' ', question).strip(),
						 "answer": str(answer).strip()})
		newdata = []
		for elem in data:
			elem["id"] = "svamp-" + str(elem["id"])
			newdata.append(elem)
		self.data = newdata
		self.clean_text()

	def load_mawps(self, fold):
		df0, df1, df2, df3, df4 = map(pd.read_csv,
									  map("../data/mawps/cv_mawps/fold{}/dev.csv".format, [0, 1, 2, 3, 4]))
		dfs = [df0, df1, df2, df3, df4]
		if fold == None:
			df = pd.concat(dfs)
		else:
			df = dfs[fold]
		data = []
		for id, (_, row) in enumerate(df.iterrows()):
			if row["Body"] != row["Body"] or row["Ques_Statement"] != row["Ques_Statement"]:
				continue
			body, question = self.impute_numbers(row["Body"], row["Ques_Statement"], row["Numbers"])
			answer = row["Answer"]
			answer = self.format_number(answer)
			data.append({"id": id,
						 "body": re.sub(' +', ' ', body).strip(),
						 "question": re.sub(' +', ' ', question).strip(),
						 "answer": str(answer).strip()})
		newdata = []
		for elem in data:
			elem["id"] = "mawps-" + str(elem["id"])
			newdata.append(elem)

		self.data = newdata
		self.clean_text()

	def load_allarith(self):
		with open("../data/all-arith/questions.json") as f:
			data = json.load(f)
		newdata = []
		for elem in data:
			elem["id"] = "allarith-" + str(elem.pop("iIndex"))
			elem["answer"] = elem.pop("lSolutions")[0]
			elem["answer"] = self.format_number(elem["answer"])
			elem["problem"] = elem.pop("sQuestion")
			newdata.append(elem)

		self.data = newdata
		self.clean_text()

	def sample_random(self):
		return random.sample(self.data, 1)[0]

	def format_number(self, number):
		if isinstance(number, float) or isinstance(number, int):
			number = str(number)
		if number[::-1].find('.') >= 5:  # if more decimal places than 5
			number = str(Fraction(float(number)).limit_denominator(100))
		else:  # remove trailing zeroes
			number = ('%f' % float(number)).rstrip('0').rstrip('.')
		return number

	def impute_numbers(self, body, question, numbers):
		numbers = numbers.split(" ")
		references = re.findall("number[0-9]", body)
		references.extend(re.findall("number[0-9]", question))
		for ref, number in zip(references, numbers):
			number = self.format_number(number)
			body = body.replace(ref, number)
			question = question.replace(ref, number)
		return body, question

	def clean_text(self):
		# this may cause problems e.g. for abbreviations, as self-illustrated by this sentence
		# can do it only for the question
		newdata = []
		for mwp in self.data:
			mwp["question"] = self._clean_text(mwp["question"])
			mwp["body"] = self._clean_text(mwp["body"])
			newdata.append(mwp)
		self.data = newdata

	def _clean_text(self, text):
		tokens = text.split(" ")
		output = tokens[0]
		#output = output[0].upper() + output[1:]
		for i in range(1, len(tokens)):
			if tokens[i] in {"n't", "'s", ",", ":", ";", ".", "?", "!", "%", '-'}:
				output += tokens[i]
			elif i > 0 and tokens[i - 1] in {'$', '-'}:
				output += tokens[i]
			elif i > 0 and (tokens[i - 1] in {'.', '?', '!'} or tokens[i - 1][-1] in {'.', '?', '!'}):
				output += ' ' + tokens[i][0].upper() + tokens[i][1:]
			else:
				output += ' ' + tokens[i]
		return output

	def remove_duplicates(self):
		problem_set = set()
		newdata = []
		for mwp in self.data:
			if "body" in mwp.keys() and "question" in mwp.keys():
				problem = mwp["body"] + " " + mwp["question"]
			else:
				problem = mwp["problem"]
			if problem not in problem_set:
				problem_set.add(problem)
				newdata.append(mwp)
		self.data = newdata

	def split_questions(self):
		"""
		Segments into body and question
		"""
		nlp = spacy.load('en_core_web_md')
		nlp.add_pipe('benepar', config={'worldmodel': 'benepar_en3'})
		newdata = []
		for mwp in tqdm(self.data):
			text = list(nlp(mwp["problem"]).sents)
			sent = text[-1]

			# consume any leading or trailing whitespace
			if str(sent)[0] == " " or str(sent)[-1] == " ":
				sent = list(nlp(str(sent).strip()).sents)[0]

			parse = sent._.parse_string
			stree = parse_stree(parse)
			subtrees = split_question(stree)
			subtrees = post_process(subtrees)

			body = " ".join([str(elem) for elem in text[:-1]])
			for subtree in subtrees[:-1]:
				body += " " + subtree.to_string()
			question = subtrees[-1].to_string()

			mwp["body"] = body.strip()
			mwp["question"] = question.strip()
			mwp.pop('problem', None)
			newdata.append(mwp)

		self.data = newdata

	def split_sentences(self):
		"""
		Segments sentences that contain multiple clauses
		"""
		nlp = spacy.load('en_core_web_md')
		nlp.add_pipe('benepar', config={'worldmodel': 'benepar_en3'})
		newdata = []
		pp_attachment_dict = {}
		for mwp in tqdm(self.data):
			text = list(nlp(mwp["body"]).sents)
			body = ""
			spans = []
			for sent in text:

				# consume any leading or trailing whitespace
				if str(sent)[0] == " " or str(sent)[-1] == " ":
					sent = list(nlp(str(sent).strip()).sents)[0]

				# handling the issue with sentences ending on a capital letter not being segmented
				if re.search(" [A-Z]\. ", str(sent)):
					temp = re.split("( [A-Z]\. )", str(sent))
					sent1 = (temp[0] + temp[1]).strip()
					sent1 = list(nlp(sent1).sents)[0]
					parse = sent1._.parse_string
					stree = parse_stree(parse)
					if is_pp_attachment_phrase(stree):
						pp_attachment_dict[mwp["id"]] = str(sent1)
					subtrees = split_top_level_clauses(stree)
					subtrees = post_process(subtrees)
					sent = list(nlp(temp[2].strip()).sents)[0]
					for subtree in subtrees:
						body += " " + subtree.to_string()
						spans.append(subtree.to_string())

				parse = sent._.parse_string
				stree = parse_stree(parse)
				if is_pp_attachment_phrase(stree):
					pp_attachment_dict[mwp["id"]] = str(sent)
				subtrees = split_top_level_clauses(stree)
				subtrees = post_process(subtrees)

				for subtree in subtrees:
					body += " " + subtree.to_string()
					spans.append(subtree.to_string())

			mwp["body"] = body.strip()
			#mwp["spans"] = dict(zip( range(len(spans)), spans ))
			mwp["spans"] = spans
			newdata.append(mwp)

		self.data = newdata
		self.pp_attachments = pp_attachment_dict