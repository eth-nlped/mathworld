from preprocessing.dataset import Dataset
import spacy
import json
from preprocessing.split_clauses import *
from tqdm import tqdm
import re
import benepar

def sample_problems(dataset, sample_size):
	out = []
	while len(out) < sample_size:
		problem = dataset.sample_random()
		if problem in out:
			continue
		else:
			out.append(problem)
	return out

def split_clauses_test(dataset, split_questions=False):
	print(dataset, "has", len(dataset), "entries")
	count1 = 0
	maxprint1 = 0
	count2 = 0
	maxprint2 = 100
	nlp = spacy.load('en_core_web_md')
	nlp.add_pipe('benepar', config={'worldmodel': 'benepar_en3'})
	# try:
	#	_create_unverified_https_context = ssl._create_unverified_context
	# except AttributeError:
	#	pass
	# else:
	#	ssl._create_default_https_context = _create_unverified_https_context
	# benepar.download()
	for mwp in tqdm(dataset):
		if "body" in mwp.keys():
			text = list(nlp(mwp["body"]).sents)
		elif "question" in mwp.keys():
			text = list(nlp(mwp["question"]).sents)
		for sent in text:

			# consume any leading or trailing whitespace
			if str(sent)[0] == " " or str(sent)[-1] == " ":
				sent = list(nlp(str(sent).strip()).sents)[0]

			# split question for gsm8k
			if split_questions and sent == text[-1]:
				parse = sent._.parse_string
				stree = parse_stree(parse)
				subtrees = split_question(stree)
				subtrees = post_process(subtrees)

				if "," in str(sent) and len(subtrees) == 1:
					print("\ndid not split the question '" + str(sent) +"'")

				if len(subtrees) != 1:
					count2 += 1
					if count2 < maxprint2:
						print("\nsplit the question '" + str(sent) + "' into: ")
						for subtree in subtrees:
							print("  " + subtree.to_string())
				continue

			# handling the issue with sentences ending on a capital letter not being segmented
			if re.search(" [A-Z]\. ", str(sent)):
				temp = re.split("( [A-Z]\. )", str(sent))
				sent1 = (temp[0] + temp[1]).strip()
				sent1 = list(nlp(sent1).sents)[0]
				parse = sent1._.parse_string
				stree = parse_stree(parse)
				subtrees = split_top_level_clauses(stree)
				subtrees = post_process(subtrees)
				sent = list(nlp(temp[2].strip()).sents)[0]
				if len(subtrees) != 1:
					count1 += 1
					if count1 < maxprint1:
						print("\nsplit the sentence '" + str(sent) + "' into: ")
						for subtree in subtrees:
							print("  " + subtree.to_string())

			parse = sent._.parse_string
			stree = parse_stree(parse)
			subtrees = split_top_level_clauses(stree)
			subtrees = post_process(subtrees)

			if len(subtrees) != 1:
				count1 += 1
				if count1 < maxprint1:
					print("\nsplit the sentence '" + str(sent) + "' into: ")
					for subtree in subtrees:
						print("  " + subtree.to_string())


	print(count1, "sentences were split")
	print(count2, "questions were split")

def get_output(dataset):
	"""
	Takes a list of mwps and returns in json format that are given as input to annotators
	"""
	out1 = []
	for mwp in dataset:
		out2 = {}
		out2["id"] = mwp["id"]
		out2["problem"] = mwp["body"] + " " + mwp["question"]
		out2["spans"] = mwp["spans"]
		out2["question"] = mwp["question"]
		out2["answer"] = mwp["answer"]
		out1.append(out2)
	return out1

def split_questions(dataset):
	"""
	Segments into body and question
	"""
	nlp = spacy.load('en_core_web_md')
	nlp.add_pipe('benepar', config={'worldmodel': 'benepar_en3'})
	newdata = []
	for mwp in tqdm(dataset):
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

	return newdata

def split_sentences(dataset):
	"""
	Segments sentences that contain multiple clauses
	"""
	nlp = spacy.load('en_core_web_md')
	nlp.add_pipe('benepar', config={'worldmodel': 'benepar_en3'})
	newdata = []
	pp_attachment_dict = {}
	for mwp in tqdm(dataset):
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

	return newdata, pp_attachment_dict