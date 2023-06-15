from worldmodel.container import Container
from worldmodel.relation import *
from worldmodel.tuple import EntityTuple
from worldmodel.state import State
from utils import viz_helper

class MWP:
	# a series of incremental states

	def __init__(self, problem_id, body, question, spans, answer = None):

		# problem id
		if not isinstance(problem_id, str):
			raise TypeError("problem id must be str")
		self.id = problem_id

		# body and question
		if not isinstance(body, str):
			raise TypeError("body must be str")
		if not isinstance(question, str):
			raise TypeError("question must be str")
		self.body = body
		self.question = question

		# spans
		if not isinstance(spans, list):
			raise TypeError("spans must be list")
		self.spans = spans
		if self.question not in self.spans:
			self.spans.append(self.question)

		# number of states in MWP
		self.num_states = len(self.spans)

		# internal counter variable for number of parsed states
		self.counter = -1

		# answer
		if not isinstance(answer, float) and not isinstance(answer, int) \
				and answer is not None:
			raise TypeError("answer must be int, float or None")
		self.answer = answer

		# incremental states
		self.states = {}

		# final state
		self.final = None

		# parsed with world for each span
		self.parsed = False

		# parsed + ref in question container
		self.determined = False

		# solved, the last state holds an answer to the MWP
		self.solved = False

		# metadata: comments, flagged, background knowledge, low confidence in annotation
		self.metadata = {}

	def __eq__(self, other):
		# note: problem id and metadata excluded
		# this is stronger than strong equivalence since also the ids must match
		return isinstance(other, MWP) and self.spans == other.spans and self.answer == other.answer \
			   and self.states == other.states and self.parsed == other.parsed \
			   and self.determined == other.determined and self.solved == other.solved

	def update_inner_state(self):
		"""
		update parsed, determined and solved
		"""
		if set(self.states.keys()) == set(range(self.num_states)):
			self.parsed = True

			last_state = self.states[self.num_states-1]
			if last_state.has_ref():
				self.determined = True

				if last_state.has_answer():
					self.solved = True

	def add_state(self, state):
		"""
		add world worldmodel for some span. world models are added incrementally with each span
		"""
		if not isinstance(state, State):
			raise TypeError("state must be of type State")
		if self.parsed:
			raise ValueError("all states already added")
		self.counter += 1
		self.states[self.counter] = state
		if self.counter == self.num_states - 1:
			self.final = state
		self.update_inner_state()

	def update_state(self, state, i):
		"""
		update state at position i with the input state
		"""
		self.states[i] = state
		if i == self.num_states -1:
			self.final = state

	def get_current_state(self):
		"""
		return the most recent parsed state
		should be useful when parsing
		"""
		return self.states[self.counter]

	def get_current_state_sequence(self):
		"""
		return current state in sequence form
		"""
		return self.states[self.counter].to_sequence()

	def has_answer(self):
		"""
		return whether last state holds an answer
		"""
		return self.states[self.num_states-1].has_answer()

	def get_answer(self):
		"""
		return answer if exists in last state
		"""
		answer = self.states[self.num_states-1].get_answer()
		self.answer = answer
		return answer

	def set_answer(self, answer):
		"""
		set answer in the last state
		"""
		self.states[self.num_states-1].set_answer(answer)
		self.answer = answer
		self.final = self.states[self.num_states-1]
		self.update_inner_state()

	def has_reference(self):
		"""
		return whether the last state holds a ref expression/variable
		"""
		return self.states[self.num_states - 1].has_ref()

	def get_reference(self):
		"""
		get ref in the last state
		"""
		return self.states[self.num_states - 1].get_ref()

	def set_reference(self, reference):
		"""
		set ref in the last state
		"""
		self.states[self.num_states - 1].set_ref(reference)
		self.final = self.states[self.num_states - 1]
		self.update_inner_state()

	def get_complete_state(self):
		"""
		return world worldmodel corresponding to question span, i.e., the full world worldmodel graph
		"""
		#if not self.parsed:
		#	raise ValueError("world worldmodel not yet parsed")
		#return self.states[-1]
		return self.final

	def set_complete_state(self, state):
		"""
		sets the world worldmodel corresponding to question span, i.e., the full world worldmodel graph
		needed if we parse the full text all at once
		"""
		self.final = state

	def add_metadata(self, metadata):
		"""
		adds metadata stemming from annotation
		"""
		if not isinstance(metadata, dict):
			raise ValueError("metadata must be a dictionary")
		for key, value in metadata.items():
			self.metadata[key] = value

	def compute_diff(self, i, sequence = False, training = True):
		"""
		compute the difference between the state at position i with state at position i-1
		if i = 0, return the difference
		difference is defined as set of containers and set of relations that are in state[i] but not state[i-1]
		edge case: sometimes quantities are updated from variable to number => return whole container/relation
		set sequence to True to return a string representation
		if training is true, we give the sequence representation used as training data. For this, we exclude containers
		without an explicit quantity if they occur together with a relation that is not part-whole, and only give
		one of the two dual transfer edges
		this is so to reduce the burden of the worldmodel to predict logical forms not present in text. These will instead
		be added by the graph update function
		for ref variable, add corresponding container/relation logical form if not already existent if at last state
		"""

		def _linearize_partwhole(relations, max_id):
			"""
			Part-wholes are linearized in conjunction due to their dependency and to compress the logical form
			This function handles that linearization as a special case
			"""
			partwholes = {j:r for j,r in relations.items() if r.type=="part-whole"}
			out = " part ( "
			whole = list(partwholes.values())[0].get_whole()
			out += f"{whole.label} , {whole.tuple.entity} , {whole.tuple.attribute} , {whole.tuple.unit} "
			for j in range(0, max_id):
				if j in partwholes.keys():
					part = partwholes[j].get_part()
					out += f", {part.label} , {part.tuple.entity} , {part.tuple.attribute} , {part.tuple.unit} "
			out += ")"
			return out

		if i not in range(0, self.num_states):
			raise ValueError("invalid index")
		elif i == 0:
			if sequence:
				return self.states[i].to_sequence(training)
			else:
				return self.states[i].containers, self.states[i].relations
		else:
			containers = self.states[i].containers
			containers_prev = list(self.states[i-1].containers.values())
			container_diff = {j:cont for j,cont in containers.items() if cont not in containers_prev}
			relations = self.states[i].relations
			relations_prev = list(self.states[i - 1].relations.values())
			relation_diff = {j:rel for j,rel in relations.items() if rel not in relations_prev}
			# check if has part-whole
			part_whole = True if any([r.type == "part-whole" for r in relation_diff.values()]) else False
			if part_whole: # define indicator of whether the part-whole relation has been linearized
				part_whole_linearized = False
			if sequence:
				out = ""
				max_id = self.states[i].get_incremented_id()
				for j in range(0, max_id):
					if j in container_diff.keys():
						if training and len(relation_diff) > 0 and container_diff[j].is_variable() and not part_whole:
							continue
						else:
							out += " " + str(container_diff[j])

					elif j in relation_diff.keys():
						if training and str(relation_diff[j]) in out and relation_diff[j].type == "transfer":
							continue
						else:
							if relation_diff[j].type != "part-whole":
								out += " " + str(relation_diff[j])
							else:
								if not part_whole_linearized:
									out += _linearize_partwhole(relation_diff, max_id)
									part_whole_linearized = True
								else:
									continue

				if i >= self.num_states - 1: # if at question, add logical form for reference variable
					try:
						ref = self.get_reference()
						ref_holders = [c for c in containers.values() if c.quantity.get_value() == ref]
						ref_holders += [r for r in relations.values() if r.quantity.get_value() == ref]
						for x in ref_holders:
							if str(x) not in out:
								out += " " + str(x)
					except:
						print("no ref variable")

				return out.strip()
			else:
				container_diff = {cont.id:cont for cont in container_diff.values()}
				relation_diff = {rel.id: rel for rel in relation_diff.values()}
				return container_diff, relation_diff

	def to_sequence(self):
		"""
		return sequential representation of state
		for shift-reduce parsing / seq2seq learning
		computes diffs between every state and concatenate
		"""
		out = ""
		for i in self.states.keys():
			out += self.compute_diff(i, True) + "\n"
		return out

	def visualize(self, i:int = None):
		"""
		visualize state at position i
		"""
		state = self.final
		if i is not None:
			state = self.states[i]
		viz_helper.visualize_mwp_state(state, mwp_name=self.id, show_plot=True)

	def to_smatch_rep_topology(self):
		"""
		exports a file with the mwp in the format required to compute smatch
		this representation only considers topology
		"""
		out = f"# {self.id}\n"
		visited = set()
		wm = self.get_complete_state()
		for rel in wm.relations.values():
			out += f"(x{rel.id} / {rel.type}\n"
			sid = rel.source.id
			out += "      :source {}".format(f"(x{sid} / container)\n" if sid not in visited else f"x{sid}\n")
			visited.add(sid)
			tid = rel.target.id
			out += "      :destination {}".format(f"(x{tid} / container)\n" if tid not in visited else f"x{tid}\n")
			visited.add(tid)
			out += ")\n"
		container_ids = {elem for elem in wm.containers.keys()}
		diff = container_ids - visited
		for cid in diff:
			out += f"(x{cid} / container)\n"
		out += "\n"
		return out

	def to_smatch_rep_full(self):
		"""
		exports a file with the mwp in the format required to compute smatch
		this representation considers the full semantics of the mwp
		"""

		def get_container_str(container, id):
			out = f"(x{id} / container\n"
			out += f'      :name (n{id} / name :op1 "{container.label}")\n'
			out += f'      :quant {container.quantity.get_value()}\n'
			out += f'      :ARG0 (e{id} / {container.tuple.entity})\n'
			if container.tuple.attribute is not None:
				out += f'      :ARG1 (a{id} / {container.tuple.attribute})\n'
			if container.tuple.unit is not None:
				out += f'      :ARG2 (u{id} / {container.tuple.unit})\n'
			out += ")\n"
			return out

		out = f"# {self.id}\n"
		visited = set()
		wm = self.get_complete_state()
		for rel in wm.relations.values():
			rid = rel.id
			out += f"(x{rid} / {rel.type}\n"

			sid = rel.source.id
			sout = get_container_str(wm.containers[sid], sid) if sid not in visited else f"x{sid}\n"
			out += f"      :source {sout}"
			visited.add(sid)

			tid = rel.target.id
			tout = get_container_str(wm.containers[tid], tid) if tid not in visited else f"x{tid}\n"
			out += f"      :destination {tout}"
			visited.add(tid)

			if rel.type == "part-whole":
				continue
			else:
				out += f"      :quant {rel.quantity.get_value()}\n"

				if rel.type == "transfer":
					out += f'      :ARG0 (e{rid} / {rel.tuple.entity})\n'
					if rel.tuple.attribute is not None:
						out += f'      :ARG1 (a{rid} / {rel.tuple.attribute})\n'
					if rel.tuple.unit is not None:
						out += f'      :ARG2 (u{rid} / {rel.tuple.unit})\n'
					if rel.recipient is not None:
						out += f'      :ARG3 (rec{rid} / {rel.recipient})\n'
					if rel.sender is not None:
						out += f'      :ARG4 (sen{rid} / {rel.sender})\n'

				elif rel.type == "rate":
					out += f'      :ARG0 (enum{rid} / {rel.tuple_num.entity})\n'
					if rel.tuple_num.attribute is not None:
						out += f'      :ARG1 (anum{rid} / {rel.tuple_num.attribute})\n'
					if rel.tuple_num.unit is not None:
						out += f'      :ARG2 (unum{rid} / {rel.tuple_num.unit})\n'
					out += f'      :ARG3 (eden{rid} / {rel.tuple_den.entity})\n'
					if rel.tuple_den.attribute is not None:
						out += f'      :ARG4 (aden{rid} / {rel.tuple_den.attribute})\n'
					if rel.tuple_den.unit is not None:
						out += f'      :ARG5 (uden{rid} / {rel.tuple_den.unit})\n'

				elif rel.type in ["explicit-add", "difference"] or rel.type in ["explicit-times", "explicit"]:
					out += f'      :ARG0 (er{rid} / {rel.res_tuple.entity})\n'
					if rel.res_tuple.attribute is not None:
						out += f'      :ARG1 (ar{rid} / {rel.res_tuple.attribute})\n'
					if rel.res_tuple.unit is not None:
						out += f'      :ARG2 (ur{rid} / {rel.res_tuple.unit})\n'
					out += f'      :ARG3 (ea{rid} / {rel.arg_tuple.entity})\n'
					if rel.arg_tuple.attribute is not None:
						out += f'      :ARG4 (aa{rid} / {rel.arg_tuple.attribute})\n'
					if rel.arg_tuple.unit is not None:
						out += f'      :ARG5 (ua{rid} / {rel.arg_tuple.unit})\n'
					if rel.result is not None:
						out += f'      :ARG6 (res{rid} / {rel.result})\n'
					if rel.argument is not None:
						out += f'      :ARG7 (arg{rid} / {rel.argument})\n'
			out += ")\n"

		container_ids = {elem for elem in wm.containers.keys()}
		diff = container_ids - visited
		for cid in diff:
			out += get_container_str(wm.containers[cid], cid)
		out += "\n"
		return out




