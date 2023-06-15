import random
from worldmodel import loader
from experiments.probing import helpers

C_QUESTS = {1: {"Q": 'How many {attr}{ent}s does {label} have?', "A": '{quant}'}, ## add now
           #-1: {"Q": 'How many {label}s does {attr} {ent} have?', "A": 'Unknown'},
            2: {"Q": 'What is the amount of {attr}{ent}s associated with {label}?', "A": '{quant}'}}
           #-2: {"Q": 'What is the amount of {label}s associated with {attr} {ent}?', "A": 'Unknown'},
           # 3: {"Q": 'What is known about {label}?', "A": 'entity: {ent}, attribute: {attr}, quantity: {quant}'}}


def container_quest(state, quest_id=1, graph_id=1):
    ## 1 –– select container
    ## special treatment of containers involved in transfer relations
    trans_source, trans_target = [], []
    for r_id, r in state.relations.items():
        if r.type == "transfer":
            trans_source.append(r.source.id)
            trans_target.append(r.target.id)

    c_list = list()
    for c_id, c in state.containers.items():
        if c.id in trans_source or c.id in trans_target: ## only consider containers involved in transfer
            if c.id == max(trans_source + trans_target): ## if container is the last one
                c_list.append(c.id)
        else: ## consider all other containers
            c_list.append(c.id)

    if isinstance(graph_id, int):
        if graph_id not in c_list: ## change invalid graph_id
            graph_id = None

    if not isinstance(graph_id, int):
        graph_id = random.choice(c_list)

    c = state.containers[graph_id]
    label, quant, ent, attr = c.label, c.quantity, c.tuple.entity, c.tuple.attribute
    quant = helpers.resolve_ref(state, quant)

    ## 2 –– select question
    if not isinstance(quest_id, int):
        quest_id = list(C_QUESTS.keys())[random.randint(0, len(C_QUESTS)-1)]

    if not isinstance(attr, str):
        attr = ""
    else:
        attr += " "

    q = C_QUESTS[quest_id]["Q"].format(quant=quant, attr=attr, ent=ent, label=label)
    a = C_QUESTS[quest_id]["A"].format(quant=quant, attr=attr, ent=ent, label=label)
    return q, a



R_QUESTS = {"transfer": {       "standard": {"Q": 'How many {ent}s does {sour} transfer to {targ}?', "A": '{quant}'},
                                "sour_only": {"Q": 'How many {ent}s are transferred from {sour}?', "A": '{quant}'},
                                "targ_only": {"Q": 'How many {ent}s are transferred to {targ}?', "A": '{quant}'}},
            "explicit-add":{    "standard": {"Q": 'How many more {ent}s does {targ} have than {sour}?', "A": '{quant}'},
                                ## entity missing, two different labels, svamp-512, Svamp-75
                                "standard": {"Q": 'What is the difference between {sour} and {targ}?', "A": '{quant}'}},
            "explicit-times":{  "standard": {"Q": 'How much more {ent} does {sour} have than {targ}?', "A": '{quant}'}},
            "rate":{            "standard": {"Q": 'How many {ent} does {targ} have per {sour}?', "A": '{quant}'}},
            "part-whole":{      "standard": {"Q": 'How many {sour} are part of {targ}?', "A": '{quant}'}}}


def relation_quest(state, graph_id=1):

    if not isinstance(graph_id, int):
        graph_id = random.choice(list(state.relations.keys()))
    if graph_id in state.relations.keys():
        r = state.relations[graph_id]
        sub_type = "standard"
        if r.type == "transfer":
            quant, sour, targ, ent = r.get_value(), r.sender, r.recipient, r.tuple.entity
            if not isinstance(sour, str): ## if sender == None
                sub_type = "targ_only"
            elif not isinstance(targ, str):  ## if sender == None
                sub_type = "sour_only"
        if r.type in ["explicit-add", "difference"] or r.type in ["explicit-times", "explicit"]:
            quant, sour, targ, ent = r.quantity.num, r.source.label, r.target.label, r.tuple.entity
        if r.type == "part-whole":
            quant, ent = r.source.quantity.get_value(), None
            sour = f"{r.source.tuple.attribute} {r.source.tuple.entity}"
            targ = f"{r.target.tuple.entity}"
        if r.type == "rate":  #r.source.tuple.entity, r.target.tuple.entity
            quant, sour, targ, ent = r.quantity.num, r.tuple_num.entity, r.target.label, r.tuple_num.entity

    quant = helpers.resolve_ref(state, quant)

    q = R_QUESTS[r.type][sub_type]["Q"].format(quant=quant, sour=sour, targ=targ, ent=ent)
    a = R_QUESTS[r.type][sub_type]["A"].format(quant=quant, sour=sour, targ=targ, ent=ent)
    return q, a