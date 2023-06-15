from dotenv import load_dotenv
import os
from pathlib import Path
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import pandas as pd
import csv

dir = os.path.dirname(__file__)
env_path = os.path.join(dir, "../.env")
dotenv_path = Path(env_path) ## need a ".env file" with the following line: OPEN_API_KEY=<KEY>
load_dotenv(dotenv_path=dotenv_path)
openai.api_key = os.getenv("OPEN_API_KEY")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def chat_generator(messages, model_id="gpt-3.5-turbo-0301"):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=messages,
		temperature=0,
		max_tokens=200,
		top_p=1,
		frequency_penalty=0.0,
		presence_penalty=0,
		stop=["\n"]
    )
    messages.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return response

seed = 5

model = "gpt-3.5-turbo-0301"
temperature = 0
presence_penalty = 0.0
system_message = {"role": "system", "content": "Your task is to generate a new math story problem based on a sequence of logical forms. You are provided with a few examples below."}

with open('prompt_examples_generation.txt', 'r') as f:
	prompt_examples = f.read()

mawps = pd.read_csv("../../output_files/splits/mawps-test.csv")
asdiv = pd.read_csv("../../output_files/splits/asdiv-test.csv")
svamp = pd.read_csv("../../output_files/splits/svamp-test.csv")
datasets = [asdiv, svamp, mawps]
for dataset in datasets:
	generated = []
	originals = []
	problem_ids = list(set(dataset["problem_id"]))
	for pid in problem_ids:
		prompt_suffix = "logical form: " + "; ".join([str(lf) for lf in list(dataset[dataset["problem_id"] == pid]["target"])]) + "\n" + "math story problem: "
		prompt = {"role": "user", "content": prompt_examples + prompt_suffix}

		gen = chat_generator([system_message, prompt])
		generated.append(gen["choices"][0].message.content.replace(";", ""))

		original = " ".join([str(lf) for lf in list(dataset[dataset["problem_id"] == pid]["source"])])
		originals.append(original)

	print("dataset finished")
	fields = ['problem_id', 'original', 'generated']
	out_test = [{"problem_id": problem_ids[i],
				 "original": originals[i],
				 "generated": generated[i]} for i in range(len(generated))]

	with open(
			f'generations/{list(dataset["problem_id"])[0][:5]}/{model}-prompt50-{temperature}-{presence_penalty}-nospacing-test.csv',
			'w', newline='') as file:
		writer = csv.DictWriter(file, fieldnames=fields)
		writer.writeheader()
		writer.writerows(out_test)
