from worldmodel import loader
from experiments.probing import taskBuilder


def create_examples(mwps, n_exampl=1, state_id=None, quest_id=None, graph_type=None, incl_answer=True, incl_premise=True):

    if not isinstance(mwps, list):
        mwps = [mwps]

    ## 2 string building___________
    examples = ""
    answers = []

    for i, mwp in enumerate(mwps):
        for j in range(0, n_exampl):
            task, a = taskBuilder.create_task(mwp, incl_answer=incl_answer, incl_premise=incl_premise, state_id=state_id, quest_id=quest_id, graph_id=None, graph_type=graph_type)
            examples += task
            answers.append(a)
    return examples[1:], answers



def create_mwp_task(mwp, n_state_exampl=1, quest_id=2, premise_mode=0, graph_type="c", instruct=False):

    ## 2 string building___________
    premise = ""
    prompt = ""
    if instruct:
        prompt += f"answer question: "
    state_ids = list(mwp.states.keys())

    if premise_mode > 0:
        for i, state_id in enumerate(state_ids[:-1]):
            if premise_mode == 1:
                premise = mwp.states[state_id].span + " "
            elif premise_mode >= 2:
                premise += mwp.states[state_id].span + " "

            prompt += premise + "\n"
            for j in range(0, n_state_exampl):
                task, a = taskBuilder.create_task(mwp, incl_answer=True, incl_premise=False, state_id=state_id, quest_id=quest_id, graph_id=None, graph_type=graph_type)
                prompt += task

    elif premise_mode == 0:
        prompt += mwp.body + "\n"
        for j in range(0, n_state_exampl):
            task, a = taskBuilder.create_task(mwp, incl_answer=True, incl_premise=False, quest_id=quest_id, graph_id=None, graph_type=graph_type)
            prompt += task

    if premise_mode >= 3:
        prompt += mwp.body + "\n"
    prompt += f"Q: {mwp.question}\nA: "
    prompt_answer = mwp.get_answer()
    return prompt, prompt_answer