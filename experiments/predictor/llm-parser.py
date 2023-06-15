from experiments.probing.probingModel import OpenAI
import pandas as pd
import csv
# could insert coref preprocessing here
from tqdm import tqdm
import numpy as np

def preprocess(text):
	return "" if text in [np.nan, "none"] else str(text)

seed = 5

model = "code-davinci-002"
temperature = 0
presence_penalty = 0.0
lm = OpenAI(model=model, temperature=temperature, presence_penalty=presence_penalty)

with open('prompt_examples.txt', 'r') as f:
	prompt_examples = f.read()

# trying the following
prompt_examples = prompt_examples.replace(" ,", ",").replace(" ( ", "(").replace(" )", ")")

mawps = pd.read_csv("../../output_files/splits/mawps-test.csv")
asdiv = pd.read_csv("../../output_files/splits/asdiv-test.csv")
svamp = pd.read_csv("../../output_files/splits/svamp-test.csv")
datasets = [asdiv, svamp, mawps]
for dataset in datasets:
	preds = []
	for i, row in tqdm(dataset.iterrows()):
		prompt_suffix = ""
		prompt_suffix += "previous sentence: " + preprocess(row["prev_source"]) + "\n"
		prompt_suffix += "previous answer: " + preprocess(row["prev_target"]) + "\n"
		prompt_suffix += "sentence: " + preprocess(row["source"]) + "\n"
		prompt_suffix += "answer: "

		pred = lm.complete(prompt_examples + prompt_suffix)
		preds.append(pred)
		#time.sleep(20)
	print("dataset finished")
	fields = ['problem_id', 'true', 'pred']
	out_test = [{"problem_id": list(dataset["problem_id"])[i],
				 "true": list(dataset["target"])[i],
				 "pred": preds[i]} for i in range(len(preds))]

	with open(
			f'predictions/{list(dataset["problem_id"])[0][:5]}/{model}-prompt50-{temperature}-{presence_penalty}-nospacing-test.csv',
			'w', newline='') as file:
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(out_test)