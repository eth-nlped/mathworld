import os
import random
from sympy import Rational
from sympy.core.symbol import Symbol

from worldmodel import loader
from worldmodel.reasoner import DeterministicReasoner


def resolve_ref(state, quant):

    if quant != None:  ## avoid relations with reference
        if not isinstance(quant, int) and not isinstance(quant, Rational) and not isinstance(quant, float):
            #if quant.is_variable():  ## containers with reference
            reasoner = DeterministicReasoner(state=state, ref=str(quant))
            quant = reasoner.reason()
            if isinstance(quant, Symbol):
                if state.has_answer():
                    quant = state.get_answer()
                else:
                    #print(quant, type(quant))
                    quant = "Unknown"
    return quant


def load_example_mwps(n_mwps=4, dataset="all"):

    if dataset == "all":
        path = "../../output_files/annotation_output/all/all-annotations/"

    if dataset == "svamp-test":
        path = "../output_files/annotation_output/svamp/test/"

    ## 1 problem loading____________
    dir = os.path.dirname(__file__)
    folder = os.path.join(dir, path)
    problems = [folder + f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    if isinstance(n_mwps, int) or n_mwps == None:
        idx = list(range(0, len(problems)))
        random.shuffle(idx)
        example_mwps = dict()
        test_mwps = dict()

        if n_mwps == None:
            n_mwps = len(problems)

        for i in idx[:n_mwps]:
            try:
                mwp_id = problems[i].split("/")[-1]
                example_mwps[mwp_id] = loader.json_to_MWP(problems[i])
            except:
                pass
                #print(f"skipped problem {mwp_id}")
        for i in idx[n_mwps:]:
            try:
                mwp_id = problems[i].split("/")[-1]
                test_mwps[mwp_id] = loader.json_to_MWP(problems[i])
            except:
                pass
                #print(f"skipped problem {mwp_id}")

    elif isinstance(n_mwps, list):
        example_mwps = dict()
        test_mwps = dict()
        for p in n_mwps:
            if p in problems:
                mwp_id = p.split("/")[-1]
                example_mwps[mwp_id] = loader.json_to_MWP(p)
        for p in (problems - example_mwps):
            mwp_id = p.split("/")[-1]
            test_mwps[mwp_id] = loader.json_to_MWP(p)

    return example_mwps, test_mwps
