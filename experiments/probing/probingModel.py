from dotenv import load_dotenv
import os
from pathlib import Path
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration

class OpenAI:

  def __init__(self, model:str="text-davinci-003", temperature=0, presence_penalty=0.0):

    dir = os.path.dirname(__file__)
    env_path = os.path.join(dir, "../.env")
    dotenv_path = Path(env_path) ## need a ".env file" with the following line: OPEN_API_KEY=<KEY>
    load_dotenv(dotenv_path=dotenv_path)
    openai.api_key = os.getenv("OPEN_API_KEY")
    self.model = model # text-davinci-003, code-davinci-002 codex, text-curie-001 gpt-2 cheaper
    self.temperature=temperature
    self.presence_penalty=presence_penalty

  @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
  def complete(self, prompt: str = ""):
    response = openai.Completion.create(
      model=self.model,
      prompt=prompt,
      temperature=self.temperature,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=self.presence_penalty,
      stop=["\n"]
    )
    return response.choices[0]["text"]



class Huggingface:

  def __init__(self, model:str="gpt2"):

    if model == "gpt2":
      self.tokenizer = AutoTokenizer.from_pretrained(model)
      self.model = AutoModelForCausalLM.from_pretrained(model)

    if model == "nt5":
      self.tokenizer = T5Tokenizer.from_pretrained("nielsr/nt5-small-rc1")
      self.model = T5ForConditionalGeneration.from_pretrained("nielsr/nt5-small-rc1")

    if model == "t5":
      self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
      self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    if model == "bart":
      self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
      self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")


  def complete(self, prompt: str = "", num_beams:int=1, max_new_tokens:int=100):

    input = self.tokenizer(prompt, return_tensors="pt")
    if num_beams <= 1:
      outputs = self.model.generate(input_ids=input["input_ids"],attention_mask = input["attention_mask"], max_new_tokens=max_new_tokens,do_sample=True,pad_token_id=self.tokenizer.eos_token_id) #max_length=10
      generation_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    else:
      outputs = self.model.generate(input_ids=input["input_ids"],attention_mask = input["attention_mask"],max_new_tokens=max_new_tokens,pad_token_id=self.tokenizer.eos_token_id,num_beams=num_beams,return_dict_in_generate=False,output_scores=False,output_hidden_states=False)
      generation_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return generation_output