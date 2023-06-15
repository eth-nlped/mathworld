from worldmodel.mwp import MWP
from worldmodel.state import State
from worldmodel.container import *
from worldmodel.relation import *

from itertools import product
import random
import sympy
from sympy import Symbol, symbols
from sympy.solvers import solve
from sympy.solvers.solveset import linsolve
from sympy.parsing.sympy_parser import parse_expr

class DeterministicReasoner():

	def __init__(self, mwp = None, state = None, ref = None, commonsense = False, orient_new = False):
		if mwp is not None:
			assert mwp.determined
			self.state = mwp.get_complete_state()
		elif state is not None and ref is not None:
			state.set_ref(ref)
			self.state = state
		self.orient = orient_new

		if commonsense:
			pass

	def set_mwp(self, mwp):
		assert mwp.determined and not mwp.solved
		self.state = mwp.get_complete_state()

	def reason(self):
		"""
		apply reasoning to state
		1. orient new edges (part whole, rate, more?)
		2. get ref variable/expression
		3. get all equations associated with the problem
		4. apply recursive solver for variable/expression over equations
		"""
		# step 1
		if self.orient:
			#self.infer_rate() # TODO only if no relation already existing
			self.infer_partwhole()

		# step 2
		ref = self.state.get_ref()
		if isinstance(ref, sympy.core.symbol.Symbol):
			# this will cover the vast majority of cases
			target = [ref]
		elif isinstance(ref, sympy.Basic):
			# if state holds an expression we need to handle the case separately: need to first solve
			# for all variables in expression and then the evaluate the expression itself
			target = [sym for sym in ref.free_symbols]
		else:
			raise TypeError("ref needs to be sympy type")

		# step 3
		eqs = self.get_equations()

		# step 4
		for var in target: # this will almost always only be one
			val = self.recursive_solver(var, eqs)
			ref = ref.subs({var:val}) # sympy will automatically simplify this
		return ref

	def recursive_solver(self, target_var, equations):
		"""
		recursive algorithm to solve for target_var given a list of equations
		"""

		def _recursive_solver(target_var, visited):

			# get all equations containing target_var and not already visited
			eqs = [eq for eq in equations if eq not in visited and isinstance(eq, sympy.Basic) and target_var in eq.free_symbols]

			# sort in increasing order according to number of free symbols
			eqs = sorted(eqs, key=lambda x: len(x.free_symbols))

			for eq in eqs:

				# can solve for target_var
				if len(eq.free_symbols) == 1:
					target_val = solve(eq, target_var)[0]
					return target_val

				# can not solve for target_var
				else:
					other_vars = eq.free_symbols.difference({target_var})
					for other_var in other_vars:
						# recursion
						other_val = _recursive_solver(other_var, visited+[eq])
						# substitute symbol with value in equation
						eq = eq.subs({other_var:other_val})

					# now check if we can solve for target_var
					if len(eq.free_symbols) == 1:
						target_val = solve(eq, target_var)[0]
						return target_val

			# could not solve for target_var
			return target_var

		answer = _recursive_solver(target_var, [])
		return answer

	def infer_partwhole(self):
		"""
		infer new part-whole edges between existing containers
		"""
		for c1, c2 in product(self.state.containers.values(), repeat=2):
			# continue if equal
			if c1 == c2:
				continue

			# continue if already exists
			elif self.state.exists_relation(c1.id, c2.id, rel_type="part-whole") or \
					self.state.exists_relation(c2.id, c1.id, rel_type="part-whole"):
				continue

			else:
				if self.part_of_whole(c1, c2):
					id = self.state.get_incremented_id()
					partwhole = PartWhole(id, source=c1, target=c2)
					self.state.add_relation(partwhole)

	def part_of_whole(self, part, whole):
		"""
		return true if there is a part whole relation between part container and whole container
		Conditions:
		(i) container labels must match, entities must match, part must have an attribute, whole may
		not have an attribute
		(ii) TODO commonsense:
			container labels must match, entity in part must be a hyponym of entity in whole
		Used for reasoner to add part-whole relations not detected by parser
		"""
		cond1 = isinstance(part, Container) and isinstance(whole, Container) and part.label == whole.label \
			   and part.tuple.entity == whole.tuple.entity and part.tuple.attribute is not None \
			   and whole.tuple.attribute is None
		cond2 = isinstance(part, Container) and isinstance(whole, Container) and part.tuple.entity == whole.tuple.entity\
			   and whole.tuple.attribute == "total"
		return cond1 or cond2

	def infer_rate(self):
		"""
		infer new rate edges between existing containers
		rule: if there is already an existing rate relation between two containers CX1 and CX2,
		and there exists two containers CY1 and CY2 such that the structure(CX1)=structure(CY1) and
		structure(CX2)=structure(CY2), then infer a rate edge between CY1 and CY2
		"""
		# TODO more efficient way to do this?
		relations = list(self.state.relations.values())
		for relation in relations:
			if relation.type != "rate":
				continue
			else:
				s1 = relation.source
				t1 = relation.target
				for _, s2 in self.state.containers.items():
					if s2 != s1 and s2 != t1 and s1.equal_structure(s2):
						for _, t2 in self.state.containers.items():
							if t2 != s1 and t2 != t1 and t1.equal_structure(t2):

								# check if rate already exists between s2 and t2
								if self.state.exists_relation(s2.id, t2.id, rel_type="rate"):
									continue

								else:
									# add new rate
									id = self.state.get_incremented_id()
									quantity = relation.quantity.get_value()
									tuple_num = relation.tuple_num
									tuple_den = relation.tuple_den
									rate = Rate(id=id, source=s2, target=t2, quantity=quantity,
												tuple_num=tuple_num, tuple_den=tuple_den)
									self.state.add_relation(rate)

	def get_equations(self):
		"""
		return a list of sympy equations over world worldmodel
		equations are the expressions when set to zero
		"""

		# handle part-whole relations separately
		equations = self.get_partwhole_equations()

		# handle all other relation types
		for r in self.state.relations.values():
			if r.type == "part-whole":
				continue
			else:
				source_num, target_num, rel_num = self.get_values(r)

				if r.type == "transfer":
					if r.source.label == r.recipient and r.target.label == r.recipient:
						# if recipient is source and target, then source + transfer = target
						equations.append(source_num + rel_num - target_num)
					elif r.source.label == r.sender and r.target.label == r.sender:
						# if sender is source and target, then source - transfer = target
						equations.append(source_num - rel_num - target_num)
					else:
						raise ValueError("transfer ill-defined")

				elif r.type == "rate":
					equations.append(source_num - target_num * rel_num)

				elif r.type in ["explicit-add", "difference"]:
					equations.append(source_num + rel_num - target_num)

				elif r.type in ["explicit-times", "explicit"]:
					equations.append(source_num * rel_num - target_num)

		return equations

	def get_values(self, relation):
		"""
		Get the values (sympy or number) associated with relation
		"""
		source_num = relation.source.quantity.get_value()
		target_num = relation.target.quantity.get_value()
		relation_num = relation.quantity.get_value()
		return source_num, target_num, relation_num

	def get_partwhole_equations(self):
		"""
		get equations associated with a part-whole relations
		1) look for whole container
		2) look for parts
		3) define equation expression
		"""
		# output
		equations = []

		# set of whole container ids that have been evaluated
		wholes = set([])

		partwholes = [r for r in self.state.relations.values() if r.type == "part-whole"]

		for relation in partwholes:
			whole = relation.target
			if whole.id in wholes:
				continue

			id_quantities = [(whole.id, whole.quantity)]

			# all part-wholes oriented towards the current whole
			inner_partwholes = [r for r in partwholes if r.target == whole]
			for r in inner_partwholes:
				id_quantities.append((r.source.id, r.source.quantity))

			expr = whole.quantity.get_value()
			for _, q in id_quantities[1:]:  # skip the whole quantity
				# sympy is useful here
				expr -= q.get_value()
			equations.append(expr)

			# add to wholes
			wholes.add(whole.id)

		return equations