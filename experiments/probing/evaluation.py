import random

from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import time

from experiments.probing import helpers, taskBuilder, taskManager, probingModel


def postprocess_prediction(y_hat):
    y_hat = re.findall(r"[-+]?(?:\d*\.*\d+)", y_hat)
    if len(y_hat) >= 1:
        y_hat = float(y_hat[-1]) ## get last mention
    else:
        y_hat = -1
    return y_hat

def metric_score(y_true, y_pred):
    acc = accuracy_score(y_true.astype(float), y_pred.astype(float))
    mse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"accuracy: {acc}, mse: {mse}")


## builder________________

def single_mwp_tasks(mwps, n_state_exampl=1, premise_mode=2, graph_type=None):
    tasks = dict()
    answers = dict()
    for mwp_id, mwp in tqdm(mwps.items()):
        prompt, a = taskManager.create_mwp_task(mwp, premise_mode=premise_mode, n_state_exampl=n_state_exampl, quest_id=None, graph_type=graph_type)
        print(prompt + "\n")
        tasks[mwp_id] = prompt
        answers[mwp_id] = a
    return tasks, answers



def other_mwp_tasks(mwps, n_state_exampl=1, other_mwp_examples=0, state_id=None, quest_id=None, graph_type=None):
    tasks = dict()
    answers = dict()
    mwps_keys = set(mwps.keys())

    for mwp_id, mwp in tqdm(mwps.items()):
        prompt = ""
        if other_mwp_examples > 0:
            for i in range(0, other_mwp_examples):
                example_mwp = random.choice(list(mwps_keys - set([mwp_id])))
                task, _ = taskBuilder.create_task(mwps[example_mwp], incl_answer=True, incl_premise=True, state_id=state_id, quest_id=quest_id, graph_id=None, graph_type=graph_type)
                prompt += task[:-1]

        for i in range(0, n_state_exampl):
            task, a = taskBuilder.create_task(mwp, incl_answer=False, incl_premise=True, state_id=state_id, quest_id=quest_id, graph_id=None, graph_type=graph_type)
            try:
                a = float(a)
                prompt += task
                tasks[mwp_id + "_" + str(i)] = prompt
                answers[mwp_id + "_" + str(i)] = a
            except:
                pass
        print(prompt)
    return tasks, answers

## executor________________

def execute_mwp_tasks(lm, tasks, answers):
    Y_hat = list()
    Y_gt = list()
    for mwp_id, prompt in tqdm(tasks.items()):
        y_hat = lm.complete(prompt)
        y_hat = postprocess_prediction(y_hat)
        y_gt = answers[mwp_id]
        print(f"{mwp_id}: y_hat: {y_hat}, y_gt: {y_gt}")
        Y_hat.append(y_hat)
        Y_gt.append(y_gt)
        if isinstance(lm.model, str): ## str if openAI worldmodel
            time.sleep(2.25) ## need to sleep for 2 seconds --> 30 requests / minute
    return np.array(Y_hat), np.array(Y_gt)