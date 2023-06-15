from worldmodel.mwp import MWP
from worldmodel.state import State
import networkx as nx
from collections import Counter

def strongly_equal(mwp1, mwp2):
	"""
	Returns true if mwp1 and mwp2 match exactly: containers, relations and their arguments all match
	This function uses the fact that if two sets have the same cardinality, injection <=> surjection <=> bijection
	"""
	state1 = mwp1.get_complete_state()
	state2 = mwp2.get_complete_state()
	if len(state1.containers) != len(state2.containers):
		return False
	if len(state1.relations) != len(state2.relations):
		return False

	# check bijection for containers
	containers1 = state1.containers.values()
	containers2 = state2.containers.values()
	for container in containers1:
		if container not in containers2:
			return False

	# check bijection for relations
	relations1 = state1.relations.values()
	relations2 = state2.relations.values()
	for relation in relations1:
		if relation not in relations2:
			return False

	return True

def weakly_equal(mwp1, mwp2):
	"""
	Returns true if mwp1 and mwp2 have the same structure/topology, including the relation types
	"""
	g1 = mwp1.get_complete_state().to_adjacency()
	g1 = nx.from_numpy_matrix(g1)
	g2 = mwp2.get_complete_state().to_adjacency()
	g2 = nx.from_numpy_matrix(g2)

	# check if the two world models contain the same relation types
	# order does not matter here
	relations1 = mwp1.get_complete_state().relations.values()
	relations2 = mwp2.get_complete_state().relations.values()
	types1 = [r.type for r in relations1]
	types2 = [r.type for r in relations2]

	return Counter(types1) == Counter(types2) and nx.is_isomorphic(g1, g2)