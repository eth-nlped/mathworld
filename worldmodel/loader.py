from worldmodel.mwp import MWP
from worldmodel.state import State
from worldmodel.relation import *
from worldmodel.tuple import EntityTuple

import json
import re
from nltk import tokenize
from nltk import WordNetLemmatizer
import copy

wnl = WordNetLemmatizer()

def update_world_model(mwp, lin, enforce_vocab = False):
	"""
	update an existing mwp object with a linearization lin of next sentence (after current state of mwp)
	linearization can be incomplete or not well-formed
	this only needs to be done at inference time
	updates mwp inplace
	"""

	vocab = tokenize.word_tokenize((mwp.body + " " + mwp.question).lower())

	# keep only the well-formed parts
	if enforce_vocab:
		lin = keep_well_formed(lin, vocab)
	else:
		lin = keep_well_formed(lin)

	if not mwp.states: # first state
		state = State(problem_id=mwp.id, span=mwp.spans[0])
		next_id = 1
	else: # start from previous state and update span
		state = copy.deepcopy(mwp.get_current_state())
		state.span = mwp.spans[len(mwp.states)]
		next_id = 1 if not state.containers else max(list(state.containers.keys()) + list(state.relations.keys()))+1

	# return if lin is empty
	if lin == "":
		mwp.add_state(state)
		return

	state_vars = state.vars
	# take variable name as the smallest unused integer
	next_var = 1 if len(state_vars) == 0 else max([int(i) for i in re.findall("[0-9]+", " ".join(state_vars))]) + 1

	# split in container/relation units
	temp = lin.split(" ")
	left_indices = [i for i, e in enumerate(temp) if e == "("]
	linl = [" ".join(temp[left_indices[i] - 1:left_indices[i + 1] - 1]) for i in range(len(left_indices) - 1)]
	linl += [" ".join(temp[left_indices[-1] - 1:])]

	for i, lform in enumerate(linl):

		try:

			lform = re.split(" , | \( | \)", lform)

			if lform[0] == "container":

				if i == len(linl) - 1 and len(mwp.states) == len(mwp.spans) - 1:  # reference variable
					ref = set_reference(state, lform, next_id, next_var)
					state.set_ref(ref)

				else:

					container = Container(next_id, label = lform[1], entity = lform[3], quantity = var_format(lform[2], next_var),
										attribute = none_format(lform[4]), unit = none_format(lform[5]))

					if container.quantity.is_variable(): # quantity is variable, no update to existing container needed
						state.add_container(container)
						next_id += 1
						next_var += 1

					else:
						# match all existing containers with same structure
						cont_match = state.matching_var_container(container)
						if cont_match: # update container with largest id (rational: recency bias)
							state.update_container(cont_match.id, container.quantity.get_value())

						else: # no existing match
							state.add_container(container)
							next_id += 1
							next_var += 1

			elif lform[0] == "transfer":

				# extract arguments
				rec_label = none_format(lform[1])
				sen_label = none_format(lform[2])
				ent = lform[4]
				attr = none_format(lform[5])
				unit = none_format(lform[6])
				transfer_var = var_format(lform[3], next_var)
				if isinstance(transfer_var, str) and "x" in transfer_var:
					next_var += 1

				# add relation for recipient
				if rec_label is not None:
					rec_matches = state.matching_containers(rec_label, ent, attr, unit)

					if not rec_matches: # create both source and target
						s_cont = Container(id=next_id, label=rec_label, entity=ent, quantity=f"x{next_var}",
									   attribute=attr, unit=unit)
						state.add_container(s_cont)
						next_id += 1
						next_var += 1
						t_cont = Container(id=next_id, label=rec_label, entity=ent, quantity=f"x{next_var}",
										   attribute=attr, unit=unit)
						state.add_container(t_cont)
						next_id += 1
						next_var += 1

					elif len(rec_matches) == 1: # create target
						s_cont = rec_matches[0]
						t_cont = Container(id=next_id, label=rec_label, entity=s_cont.tuple.entity, quantity=f"x{next_var}",
										   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
						state.add_container(t_cont)
						next_id += 1
						next_var += 1

					elif len(rec_matches) > 1:
						# if one of the matches was created in this state, then that is the target and the most previous one is the source
						# most recent container is source, create new target
						if i > 0:
							if linl[i-1][:9] == "container":
								s_cont = rec_matches[1]
								t_cont = rec_matches[0]
							else:
								s_cont = rec_matches[0]
								t_cont = Container(id=next_id, label=rec_label, entity=s_cont.tuple.entity,
												   quantity=f"x{next_var}",
												   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
								state.add_container(t_cont)
						else:
							s_cont = rec_matches[0]
							t_cont = Container(id=next_id, label=rec_label, entity=s_cont.tuple.entity, quantity=f"x{next_var}",
											   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
							state.add_container(t_cont)
						next_id += 1
						next_var += 1

					rel = Transfer(id=next_id, source=s_cont, target=t_cont, quantity=transfer_var,
								   tuple=EntityTuple(ent, attr, unit), recipient=rec_label, sender=sen_label)

					if i == len(linl) - 1 and len(mwp.states) == len(mwp.spans) - 1:  # reference variable
						# check if matches existing relation, if does, then that is the ref
						matches = sorted([r for r in state.relations.values() if r.equal_structure(rel)],
										 key=lambda x: x.id, reverse=True)
						if matches:
							ref = matches[0].quantity.get_value()
						else:
							state.add_relation(rel)
							ref = rel.quantity.get_value()
							next_id += 1
						state.set_ref(ref)

					else:
						state.add_relation(rel)
						next_id += 1

				# add relation for sender
				if sen_label is not None:
					sen_matches = state.matching_containers(sen_label, ent, attr, unit)

					if not sen_matches:  # create both source and target
						s_cont = Container(id=next_id, label=sen_label, entity=ent, quantity=f"x{next_var}",
										   attribute=attr, unit=unit)
						state.add_container(s_cont)
						next_id += 1
						next_var += 1
						t_cont = Container(id=next_id, label=sen_label, entity=ent, quantity=f"x{next_var}",
										   attribute=attr, unit=unit)
						state.add_container(t_cont)
						next_id += 1
						next_var += 1

					elif len(sen_matches) == 1:  # create target
						s_cont = sen_matches[0]
						t_cont = Container(id=next_id, label=sen_label, entity=s_cont.tuple.entity, quantity=f"x{next_var}",
										   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
						state.add_container(t_cont)
						next_id += 1
						next_var += 1

					elif len(sen_matches) > 1:  # most recent container is source, create new target
						# if one of the matches was created in this state, then that is the target and the most previous one is the source
						# most recent container is source, create new target
						if i > 0:
							if linl[i - 1][:9] == "container":
								s_cont = sen_matches[1]
								t_cont = sen_matches[0]
							else:
								s_cont = sen_matches[0]
								t_cont = Container(id=next_id, label=sen_label, entity=s_cont.tuple.entity,
												   quantity=f"x{next_var}",
												   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
								state.add_container(t_cont)

						else:
							s_cont = sen_matches[0]
							t_cont = Container(id=next_id, label=sen_label, entity=s_cont.tuple.entity,
											   quantity=f"x{next_var}",
											   attribute=s_cont.tuple.attribute, unit=s_cont.tuple.unit)
							state.add_container(t_cont)
						next_id += 1
						next_var += 1

					rel = Transfer(id=next_id, source=s_cont, target=t_cont, quantity=transfer_var,
								   tuple=EntityTuple(ent, attr, unit), recipient=rec_label, sender=sen_label)

					if i == len(linl) - 1 and len(mwp.states) == len(mwp.spans) - 1:  # reference variable
						# check if matches existing relation, if does, then that is the ref
						matches = sorted([r for r in state.relations.values() if r.equal_structure(rel)],
										 key=lambda x: x.id, reverse=True)
						if matches:
							ref = matches[0].quantity.get_value()
						else:
							state.add_relation(rel)
							ref = rel.quantity.get_value()
							next_id += 1
						state.set_ref(ref)
					else:
						state.add_relation(rel)
						next_id += 1

				next_id += 1

			elif lform[0] == "rate":

				# extract arguments
				label = lform[1]
				# find source
				s_ent, s_attr, s_unit = lform[3], none_format(lform[4]), none_format(lform[5])
				s_matches = state.matching_containers(label, s_ent, s_attr, s_unit)
				# find target
				t_ent, t_attr, t_unit = lform[6], none_format(lform[7]), none_format(lform[8])
				t_matches = state.matching_containers(label, t_ent, t_attr, t_unit)

				# create source container if not match
				if not s_matches:
					s_cont = Container(id=next_id, label=label, entity=s_ent, quantity=f"x{next_var}",
									   attribute=s_attr, unit=s_unit)
					state.add_container(s_cont)
					s_matches.append(s_cont)
					next_id += 1
					next_var += 1

				# create target container if not match
				if not t_matches:
					t_cont = Container(id=next_id, label=label, entity=t_ent, quantity=f"x{next_var}",
									   attribute=t_attr, unit=t_unit)
					state.add_container(t_cont)
					t_matches.append(t_cont)
					next_id += 1
					next_var += 1

				# add a rate edge between the most recent matches
				# previously for every pair of matches but that introduced errors
				s = s_matches[0]
				t = t_matches[0]
				rel = Rate(id=next_id, source=s, target=t, quantity = var_format(lform[2], next_var),
						   tuple_num=EntityTuple(s_ent, s_attr, s_unit), tuple_den=EntityTuple(t_ent, t_attr, t_unit))

				if i == len(linl) - 1 and len(mwp.states) == len(mwp.spans) - 1:  # reference variable
					# check if matches existing relation, if does, then that is the ref
					matches = sorted([r for r in state.relations.values() if r.equal_structure(rel)],
									 key=lambda x: x.id, reverse=True)
					if matches:
						ref = matches[0].quantity.get_value()
					else:
						state.add_relation(rel)
						ref = rel.quantity.get_value()
						next_id += 1
						if rel.quantity.is_variable():
							next_var += 1
					state.set_ref(ref)
				else:
					state.add_relation(rel)
					next_id += 1
					if rel.quantity.is_variable():
						next_var += 1

			elif lform[0] == "part":
				# extract whole arguments
				lform.reverse()
				lform.pop()
				whole_label = lform.pop()
				whole_ent = lform.pop()
				whole_attr = none_format(lform.pop())
				whole_unit = none_format(lform.pop())

				whole_matches = state.matching_containers(whole_label, whole_ent, whole_attr, whole_unit)
				if whole_matches:
					t_cont = whole_matches[0]
				else:  # take most recently created container if no match
					t_cont = sorted(list(state.containers.items()), key=lambda x: x.id, reverse=True)[0]

				while len(lform) >= 4:
					# extract part arguments until lform is empty
					part_label = lform.pop()
					part_ent = lform.pop()
					part_attr = none_format(lform.pop())
					part_unit = none_format(lform.pop())

					# here all containers should already be existing
					part_matches = state.matching_containers(part_label, part_ent, part_attr, part_unit)
					# create an edge between most recent res and arg matches (rationale: if a transfer occurred we want after that)
					s_cont = part_matches[0]

					rel = PartWhole(id=next_id, source=s_cont, target=t_cont)
					state.add_relation(rel)
					next_id += 1

			elif lform[0] in ["add", "times", "difference", "explicit"]:

				# extract arguments
				res_label = lform[1]
				arg_label = lform[2]
				res_ent = lform[4]
				res_attr = none_format(lform[5])
				res_unit = none_format(lform[6])
				arg_ent = lform[7]
				arg_attr = none_format(lform[8])
				arg_unit = none_format(lform[9])

				res_matches = state.matching_containers(res_label, res_ent, res_attr, res_unit, soft=False)
				arg_matches = state.matching_containers(arg_label, arg_ent, arg_attr, arg_unit, soft=False)

				# if one (and not both) of res_matches or arg_matches is empty, do a soft match on that one
				# make sure the matches don't overlap
				if (not res_matches or not arg_matches) and res_matches:
					arg_matches = state.matching_containers(arg_label, arg_ent, arg_attr, arg_unit, soft=True)
					arg_matches = [c for c in arg_matches if c not in res_matches]
				if (not res_matches or not arg_matches) and arg_matches:
					res_matches = state.matching_containers(res_label, res_ent, res_attr, res_unit, soft=True)
					res_matches = [c for c in res_matches if c not in arg_matches]

				# create result container if not match
				if not res_matches:
					res_cont = Container(id=next_id, label=res_label, entity=res_ent, quantity=f"x{next_var}",
									   attribute=res_attr, unit=res_unit)
					state.add_container(res_cont)
					res_matches.append(res_cont)
					next_id += 1
					next_var += 1

				# create argument container if not match
				if not arg_matches:
					arg_cont = Container(id=next_id, label=arg_label, entity=arg_ent, quantity=f"x{next_var}",
										 attribute=arg_attr, unit=arg_unit)
					state.add_container(arg_cont)
					arg_matches.append(arg_cont)
					next_id += 1
					next_var += 1

				# create an edge between most recent res and arg matches (rational: if a transfer occurred we want after that)
				s_cont = arg_matches[0]
				t_cont = res_matches[0]
				if lform[0] == "difference":
					rel = ExplicitAdd(id=next_id, source=s_cont, target=t_cont, quantity= var_format(lform[3], next_var),
									  res_tuple=EntityTuple(res_ent, res_attr, res_unit), arg_tuple=EntityTuple(arg_ent, arg_attr, arg_unit),
									  argument=arg_label, result=res_label)
				elif lform[0] == "explicit":
					rel = ExplicitTimes(id=next_id, source=s_cont, target=t_cont, quantity=var_format(lform[3], next_var),
										res_tuple=EntityTuple(res_ent, res_attr, res_unit), arg_tuple=EntityTuple(arg_ent, arg_attr, arg_unit),
										argument=arg_label, result=res_label)

				if i == len(linl) - 1 and len(mwp.states) == len(mwp.spans) - 1:  # reference variable
					# check if matches existing relation, if does, then that is the ref
					matches = sorted([r for r in state.relations.values() if r.equal_structure(rel)],
									 key=lambda x: x.id, reverse=True)
					if matches:
						ref = matches[0].quantity.get_value()
					else:
						state.add_relation(rel)
						ref = rel.quantity.get_value()
						next_id += 1
						if rel.quantity.is_variable():
							next_var += 1
					state.set_ref(ref)
				else:
					state.add_relation(rel)
					next_id += 1
					if rel.quantity.is_variable():
						next_var += 1

		except TypeError:
			print("TypeError: Semantic ill-formedness")
		except ValueError:
			print("ValueError: Semantic ill-formedness")
		except:
			print("Unknown error")

	if state.ref is None and len(mwp.states) == len(mwp.spans) - 1: # does not have a reference, which should indicate that the last lform was part-whole
		# set ref to last container that was added (which should be unknown part or whole container in the question) that has a variable
		c_ref = sorted([c for c in list(state.containers.values()) if c.is_variable()], key=lambda x: x.id, reverse=True)
		if c_ref:
			ref = c_ref[0].quantity.get_value()
			state.set_ref(ref)
		else: # set to most recent unknown relation var if possible
			r_ref = sorted([r for r in list(state.relations.values()) if r.type != "part-whole" and r.is_variable()],
				   key=lambda x: x.id, reverse=True)
			if r_ref:
				ref = r_ref[0].quantity.get_value()
				state.set_ref(ref)


	mwp.add_state(state)

def none_format(s):
	return s if s not in ["none", "None"] else None

def var_format(s, var_num):
	return s if s not in ["none", "None"] else f"x{var_num}"

def set_reference(state, lform, id, var):
	"""
	find ref variable that matches lform for containers
	create a container if not already existing
	"""
	ref_container = Container(id, label = lform[1], entity = lform[3], quantity = var_format(lform[2], var),
							attribute = none_format(lform[4]), unit = none_format(lform[5]))
	cont_match = state.matching_var_container(ref_container, soft=True)
	if cont_match:
		ref = cont_match.quantity.get_value()
	else: # default to most recent container ref
		c_ref = sorted([c for c in list(state.containers.values()) if c.is_variable()], key=lambda x: x.id,
					   reverse=True)
		if c_ref:
			ref = c_ref[0].quantity.get_value()
		else:  # set to most recent unknown relation var if possible
			r_ref = sorted([r for r in list(state.relations.values()) if r.type != "part-whole" and r.is_variable()],
						   key=lambda x: x.id, reverse=True)
			if r_ref:
				ref = r_ref[0].quantity.get_value()
			else:
				state.add_container(ref_container)
				ref = ref_container.quantity.get_value()
	return ref

def keep_well_formed(lin, vocab = None):
	"""
	take a linearization lin and output only the parts that are syntactically
	well-formed according to the linearization specification
	(see __str__ methods for containers and relations)
	# instead of general regular expressions can insert the vocabulary from the problem: [token1,token2,...,tokenN]
	# plus special tokens none, time, money, world
	"""
	if not isinstance(lin, str):
		return ""

	lin = lin.lower().strip()
	# required formatting
	lin.replace("none.", "none,")
	lin = re.sub('([,!?()])', r' \1 ', lin)
	lin = re.sub('\s{2,}', ' ', lin)

	if vocab:
		vocab += [wnl.lemmatize(word) for word in vocab]
		special_tokens = ["none", "world", "money", "time", "occasion"]
		vocab += special_tokens
		#vocab = [word for word in vocab if word.isalpha()] # remove numerics (but it removes also tokens like mrs.)
		constr = "[(" + "|".join([str(elem) for elem in vocab]) + ")\s]+"
	else:
		constr = "[a-zA-Z\s\.']+"

	# regex for integers, decimals and fractions
	n_constr = "([0-9]*[.|/]?[0-9]*|none)"

	# define well-formed formats for logical forms
	container_form = "container ( {} , {} , {} , {} , {} )".format(constr, n_constr, constr, constr, constr)
	transfer_form = "transfer ( {} , {} , {} , {} , {} , {} )".format(constr, constr, n_constr, constr, constr, constr)
	rate_form = "rate ( {} , {} , {} , {} , {} , {} , {} , {} )".format(constr, n_constr, constr, constr, constr, constr, constr, constr)
	partwhole_form1 = ("part (" + " {x} ," * 7 + " {x} )").format(x=constr) # should be able to represent these with the same regex
	partwhole_form2 = ("part (" + " {x} ," * 11 + " {x} )").format(x=constr)
	partwhole_form3 = ("part (" + " {x} ," * 15 + " {x} )").format(x=constr)
	partwhole_form4 = ("part (" + " {x} ," * 19 + " {x} )").format(x=constr)
	partwhole_form5 = ("part (" + " {x} ," * 23 + " {x} )").format(x=constr)
	explicitadd_form = "difference ( {} , {} , {} , {} , {} , {} , {} , {} , {} )".format(constr, constr, n_constr, constr, constr, constr, constr, constr, constr)
	explicittimes_form = "explicit ( {} , {} , {} , {} , {} , {} , {} , {} , {} )".format(constr, constr, n_constr, constr, constr, constr, constr, constr, constr)
	matched = re.findall(
		f"({container_form}|{transfer_form}|{rate_form}|{partwhole_form1}|{partwhole_form2}|{partwhole_form3}|{partwhole_form4}|{partwhole_form5}|{explicitadd_form}|{explicittimes_form})".replace(' ( ', ' [(] ').replace(' )', ' [)]'),
		lin)
	out = ""
	for i in matched:
		out += i[0] + " "
	out = out.strip()
	return out

def json_to_MWP(json_path):
	"""
	input an annotation in json format and output an MWP object
	"""

	with open(json_path, "r") as f:
		data = json.load(f)

	# initialize MWP object
	problem_id = data[0]["graph"]["id"]
	body = ""
	spans = []
	for i, graph in enumerate(data):
		text_span = graph["graph"]["metadata"]["text span"]
		spans.append(text_span)
		if i != len(data)-1:
			# do not add question to the body
			body += text_span + " "
	body = body.strip()
	question = data[len(data) - 1]["graph"]["metadata"]["text span"]
	mwp = MWP(problem_id=problem_id, body=body, question=question, spans=spans)

	# add states
	for graph in data:
		state = parse_into_state(graph)
		mwp.add_state(state)

	# add metadata
	metadata = data[-1]["graph"]["metadata"]
	metadata.pop("text span", None)
	mwp.add_metadata(metadata)

	return mwp

def parse_into_state(state_dict):
	"""
	parse into state based on state_dict from json
	"""
	graph = state_dict["graph"]
	state = State(problem_id=graph["id"], span=graph["metadata"]["text span"])

	# add containers
	for container_id, attributes in graph["nodes"].items():
		attributes = unpack_container(attributes)
		if container_id == "CQ": # or attributes["reference"] is not None:
			state.set_answer(attributes["quantity"])
			state.set_ref(attributes["reference"])
		else:
			attributes.pop("reference", None)
			container = Container(id=container_id, **attributes)
			state.add_container(container)

	# add relations
	for relation in graph["edges"]:
		relation_id = relation["id"]
		source = state.containers[int(relation["source"])]
		target = state.containers[int(relation["target"])]
		if source == "undefined" or target == "undefined":
			print("Error! container undefined")
		type = relation["relation"]
		attributes = unpack_relation(type, relation["metadata"])

		if type == "transfer":
			relation = Transfer(id=relation_id, source=source, target=target, **attributes)
		elif type == "rate":
			relation = Rate(id=relation_id, source=source, target=target, **attributes)
		elif type == "part-whole":
			relation = PartWhole(id=relation_id, source=source, target=target)
		elif type in ["explicit-add", "explicit-times", "difference", "explicit"]:
			# here we adapt to the new argument structure

			if attributes["result"] == attributes["argument"]: # need to take tuples from the containers
				attributes["res_tuple"] = target.tuple
				attributes["arg_tuple"] = source.tuple
				del attributes["tuple"]

			else: # we have the same tuples for both result and argument
				attributes["res_tuple"] = attributes["tuple"]
				attributes["arg_tuple"] = attributes["tuple"]
				del attributes["tuple"]

			if type in ["explicit-add", "difference"]:
				relation = ExplicitAdd(id=relation_id, source=source, target=target, **attributes)
			elif type in ["explicit-times", "explicit"]:
				relation = ExplicitTimes(id=relation_id, source=source, target=target, **attributes)

		state.add_relation(relation)

	return state

def unpack_container(node):
	"""
	prepare format for container constructor
	"""
	out = {}
	out["label"] = node["label"]
	metadata = node["metadata"]
	for key, value in metadata.items():
		if value in ["-", "", " "]:
			# optional argument left without value
			out[key] = None
		else:
			out[key] = value
	return out

def unpack_relation(type, edge_attributes):
	"""
	prepare format for relation constructor
	"""
	out = {}
	if type == "transfer":
		out["quantity"] = edge_attributes["X1"]
		tuple = edge_attributes["X2"]
		entity = tuple[0]
		attribute = tuple[1] if tuple[1] not in ["-", "", " "] else None
		unit = tuple[2] if tuple[2] not in ["-", "", " "] else None
		out["tuple"] = EntityTuple(entity=entity, attribute=attribute, unit=unit)
		out["recipient"] = edge_attributes["X3"] if edge_attributes["X3"] not in ["-", "", " "] else None
		out["sender"] = edge_attributes["X4"] if edge_attributes["X4"] not in ["-", "", " "] else None

	elif type == "rate":
		out["quantity"] = edge_attributes["X1"]
		tuple_num = EntityTuple(entity=edge_attributes["X2"][0],
								attribute=edge_attributes["X2"][1] if edge_attributes["X2"][1] not in ["-", "", " "] else None,
								unit=edge_attributes["X2"][2] if edge_attributes["X2"][2] not in ["-", "", " "] else None)
		out["tuple_num"] = tuple_num
		tuple_den = EntityTuple(entity=edge_attributes["X3"][0],
								attribute=edge_attributes["X3"][1] if edge_attributes["X3"][1] not in ["-", "", " "] else None,
								unit=edge_attributes["X3"][2] if edge_attributes["X3"][2] not in ["-", "", " "] else None)
		out["tuple_den"] = tuple_den

	elif type in ["explicit-add", "explicit-times", "difference", "explicit"]:

		out["quantity"] = edge_attributes["X1"]
		out["tuple"] = EntityTuple(entity=edge_attributes["X2"][0],
								attribute=edge_attributes["X2"][1] if edge_attributes["X2"][1] not in ["-", "", " "] else None,
								unit=edge_attributes["X2"][2] if edge_attributes["X2"][2] not in ["-", "", " "] else None)
		out["result"] = edge_attributes["X3"]
		out["argument"] = edge_attributes["X4"]

	return out
