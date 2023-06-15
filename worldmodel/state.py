from worldmodel.container import Container
from worldmodel.relation import *
from worldmodel.tuple import EntityTuple
from utils import viz_helper

import sympy
from sympy import symbols, Rational
from sympy.parsing.sympy_parser import parse_expr

import numpy as np

class State:
	# meant for each intermediate world worldmodel state up until (inclusive) a given text span

	def __init__(self, problem_id, span):

		# problem id
		if not isinstance(problem_id, str):
			raise TypeError("problem id must be str")
		self.id = problem_id

		# span
		if not isinstance(span, str):
			raise TypeError("span must be str")
		self.span = span

		# dict of containers, keys are int
		self.containers = {}

		# dict of relations, keys are int
		self.relations = {}

		# answer
		self.answer = None

		# ref variable/expression
		self.ref = None

		# variables as str
		self.vars = []

	def __eq__(self, other):
		return isinstance(other, State) and self.span == other.span and self.containers == other.containers \
			and self.relations == other.relations

	def add_container(self, container):
		if not isinstance(container, Container):
			raise TypeError("container must be of type Container")
		self.containers[container.id] = container
		# add to vars if variable
		if container.quantity.is_variable():
			self.vars.append(str(container.quantity.get_value()))

	def update_container(self, container_id, value):
		"""
		Update an existing container by setting a variable quantity to a value
		This is always done in a subsequent state. Hence, first deep copy state and then call this method
		"""
		self.containers[container_id].set_value(value)

	def set_answer(self, answer):
		"""
		set a predicted answer
		"""
		if not isinstance(answer, float) and not isinstance(answer, int) and not isinstance(answer, Rational)\
				and not isinstance(answer, str):
			raise TypeError("answer must be int, float, Rational or str")
		if self.has_answer():
			raise ValueError("answer has already been set")
		else:
			if not isinstance(answer, str):
				self.answer = answer
			elif is_fraction(answer):
				self.answer = Rational(answer)
			elif is_int(answer):
				self.answer = int(answer)
			elif is_float(answer):
				self.answer = float(answer)
			else:
				raise ValueError("invalid str")

	def has_answer(self):
		"""
		return true if state holds an answer
		"""
		if self.answer is not None:
			return True
		else:
			return False

	def get_answer(self):
		"""
		return the answer
		"""
		if not self.has_answer():
			raise ValueError("answer not known")
		else:
			return self.answer

	def set_ref(self, ref):
		"""
		set ref in question container
		"""
		if not isinstance(ref, str) and not isinstance(ref, sympy.Basic):
			raise TypeError("ref must be str, sympy object or None")

		if isinstance(ref, str):
			self.ref = parse_expr(ref)
		elif isinstance(ref, sympy.Basic):
			self.ref = ref

	def has_ref(self):
		"""
		return true if state holds a ref
		"""
		if self.ref is not None:
			return True
		else:
			return False

	def get_ref(self):
		"""
		get ref in the question container
		"""
		if not self.has_ref():
			raise ValueError("ref not known")
		else:
			return self.ref

	def add_relation(self, relation):
		if not isinstance(relation, Relation):
			raise TypeError("relation must be of type Relation")
		if not relation.source_id in self.containers.keys():
			raise ValueError("source container must exist in world worldmodel")
		if not self.containers[relation.source_id] == relation.source:
			raise ValueError("source containers must match")
		if not relation.target_id in self.containers.keys():
			raise ValueError("target container must exist in world worldmodel")
		if not self.containers[relation.target_id] == relation.target:
			raise ValueError("target containers must match")
		self.relations[relation.id] = relation
		# add to vars if variable
		if relation.type != "part-whole":
			if relation.quantity.is_variable():
				self.vars.append(str(relation.quantity.get_value()))

	def update_relation(self, relation_id, value):
		"""
		Update an existing relation by setting a variable quantity to a value
		"""
		self.relations[relation_id].set_value(value)

	def get_incremented_id(self):
		"""
		returns max(ids)+1
		or 1 if there exists no containers
		"""
		return 1 if not self.containers else max(list(self.containers.keys()) + list(self.relations.keys())) + 1

	def exists_relation(self, source_id, target_id, rel_type=None):
		"""
		Check if there exists a relation of type rel_type between source_id and target_id
		"""
		for r in self.relations.values():
			if rel_type is not None:
				if r.source.id == source_id and r.target.id == target_id and r.type == rel_type:
					return True
			else:
				if r.source.id == source_id and r.target.id == target_id:
					return True
		return False

	def matching_containers(self, label, entity, attr, unit, soft = True):
		"""
		Check if there exists any container with the input values, and return all such containers
		if soft, return any matching container only on entity if attr and unit do not match
		"""
		matches = []
		dummy_id = 100
		dummy_quantity = 100
		container = Container(dummy_id, label, entity, dummy_quantity, attr, unit)

		# collect all matches
		for i, c in self.containers.items():
			if container.equal_structure(c):
				matches.append(c)

		# softer match
		if not matches and soft:
			for i, c in self.containers.items():
				if container.equal_structure(c, soft=True):
					matches.append(c)

		# sort by most recent first
		if matches:
			return sorted(matches, key=lambda x: x.id, reverse=True)
		else:
			return matches

	def matching_containers_le(self, label, entity):
		"""
		like matching_containers but only matches on label and entity
		"""
		matches = []
		dummy_id = 100
		dummy_quantity = 100
		container = Container(dummy_id, label, entity, dummy_quantity)

		# collect all matches
		for i, c in self.containers.items():
			if c.label == container.label and c.tuple.entity == container.tuple.entity:
				matches.append(c)

		# sort by most recent first
		if matches:
			return sorted(matches, key=lambda x: x.id, reverse=True)
		else:
			return matches

	def matching_var_container(self, container, soft = False):
		"""
		Check if there exists a container with a variable that has the same structure as input container
		and return that container
		If there are multiple, return the one with the largest id (reason: assume recency bias)
		"""
		matches = []

		# collect all matches
		for i, c in self.containers.items():
			if container.equal_structure(c) and c.quantity.is_variable():
				matches.append((i, c))

		# softer match
		if not matches and soft:
			for i, c in self.containers.items():
				if container.label == c.label and container.tuple.entity == c.tuple.entity and c.quantity.is_variable():
					matches.append((i, c))

		# take most recent
		if matches:
			return sorted(matches, key=lambda x: x[0])[-1][1]
		else:
			return matches

	def to_sequence(self, train = False):
		"""
		return sequential representation of state
		follows id ordering
		for shift-reduce parsing / seq2seq learning
		if train is True we give the representation used as training data. For this, we exclude containers
		without an explicit quantity if they occur together with a relation that is not part-whole
		"""
		if len(self.containers) == 0 and len(self.relations) == 0:
			return ""
		else:
			out = ""
			max_id = self.get_incremented_id()

			# check if has part-whole
			part_whole = True if any([r.type == "part-whole" for r in self.relations.values()]) else False

			for i in range(0, max_id):
				if i in self.containers.keys():

					if train:
						if len(self.relations) > 0 and self.containers[i].is_variable() and not part_whole:
							continue
						else:
							out += " " + str(self.containers[i])

					else:
						out += " " + str(self.containers[i])

				elif i in self.relations.keys():
					if train and str(self.relations[i]) in out and self.relations[i].type == "transfer":
						continue
					else:
						out += " " + str(self.relations[i])

			return out.strip()

	def to_adjacency(self):
		"""
		gives an adjacency matrix of the graph
		"""
		adj = np.zeros([len(self.containers),len(self.containers)])

		# id2pos solves the issue of ids not having been incremented properly
		id2pos = {id: i for i, id in enumerate(list(self.containers.keys()))}
		for relation in self.relations.values():
			sid = relation.source.id
			tid = relation.target.id
			s_num = id2pos[sid]
			t_num = id2pos[tid]
			adj[s_num, t_num] = 1
		return adj

	def visualize(self):
		"""
		plot the state
		"""
		viz_helper.visualize_mwp_state(self, mwp_name=self.id, show_plot=True)