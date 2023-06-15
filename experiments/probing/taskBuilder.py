import random
from worldmodel import loader
from experiments.probing import questPhraser


def use_mwp_task(mwp):
    premise = mwp.body
    q = mwp.question
    a = str(mwp.get_answer())
    return premise, q, a


def create_task(mwp, incl_answer=True, incl_premise=True, state_id=None, quest_id=None, graph_id=None, graph_type=None):
    if state_id == -1: ## use original math problem question
        premise, q, a = use_mwp_task(mwp)
    else:
        if state_id == -2:  ## use last state before question, i.e. complete world worldmodel
            state_id = len(mwp.states) - 2
        state, premise, graph_type = select_state(mwp, state_id=state_id, graph_type=graph_type)
        q, a = phrase_quest(state, quest_id=quest_id, graph_id=graph_id, graph_type=graph_type)

    task = ""
    if len(q) > 0:
        if incl_premise:
            task += f"\n{premise}\n"
        task += f"Q: {q}\nA: "
        if incl_answer:
            task += f"{a}\n"
    return task, a


def phrase_quest(state, quest_id=1, graph_id=1, graph_type="c"):
    if graph_type == "c":
        q, a = questPhraser.container_quest(state, quest_id=quest_id, graph_id=graph_id)
    elif graph_type == "r":
        q, a = questPhraser.relation_quest(state, graph_id=graph_id)
    else:
        q, a = "", ""
    return q, a


def select_state(mwp, state_id=0, graph_type="c"):

    if not isinstance(graph_type, str): ## select graph time (container or relation)
        graph_type = ["c", "r"][random.randint(0, 1)]

    c_states = list()
    r_states = list()

    for id in list(mwp.states.keys())[:-1]:
        if len(mwp.states[id].relations) > 0:
            r_states.append(id)
        if len(mwp.states[id].containers) > 0:
            c_states.append(id)

    if isinstance(state_id, int):
        if (graph_type == "r" and state_id not in r_states):
            graph_type = "c"
        if (graph_type == "c" and state_id not in c_states):
            graph_type = None
    else:
        if graph_type == "r" and len(r_states) > 0: ## avoid graph type "r" when no relations
            state_id = random.choice(r_states)
        else:
            graph_type = "c"
            state_id = random.choice(c_states)

    state = mwp.states[state_id]
    premise = ""
    for i, span in enumerate(mwp.spans[:state_id + 1]):
        premise += span + " "
    return state, premise[:-1], graph_type