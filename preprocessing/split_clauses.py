import csv

class SyntaxNode(object):
	def __init__(self, label, children=None):
		if label == '':
			raise ValueError("SyntaxNode ERROR: Label is empty.")
		self.label = label
		self.children = children

	def to_stree(self):
		if self.children is None:
			return self.label
		return '(' + self.label + ' ' + ' '.join([child.to_stree() for child in self.children]) + ')'

	def capitalize(self):
		if self.children is None:
			self.label = self.label[0].upper() + self.label[1:]
			return
		self.children[0].capitalize()

	def to_tokens(self):
		if self.children is None:
			return [self.label]
		tokens = []
		for child in self.children:
			tokens.extend(child.to_tokens())
		return tokens

	def to_string(self):
		tokens = self.to_tokens()
		output = tokens[0]
		for i in range(1, len(tokens)):
			if tokens[i] in {"n't", "'s", ",", ":", ";", ".", "?", "!", "%", '-'}:
				output += tokens[i]
			elif i > 0 and tokens[i - 1] in {'$', '-'}:
				output += tokens[i]
			else:
				output += ' ' + tokens[i]
		return output

def do_parse_stree(input, position):
	# consume any leading whitespace
	while input[position] == ' ' and position < len(input):
		position += 1

	if input[position] == '(':
		position += 1
		space_index = input.find(' ', position)
		if space_index == -1:
			raise Exception("Invalid syntax in S-expression: Encountered nonterminal with no child nodes.")
		label = input[position:space_index]

		position = space_index + 1
		children = []
		while True:
			(child_node, position) = do_parse_stree(input, position)
			children.append(child_node)
			if input[position] == ')':
				position += 1
				break
			elif input[position] != ' ':
				raise Exception("Invalid syntax in S-expression: Expected a space separating the child nodes of a nonterminal.")
			position += 1
		return (SyntaxNode(label, children), position)

	else:
		# this is a terminal node
		space_index = input.find(' ', position)
		rparen_index = input.find(')', position)
		if space_index == -1:
			if rparen_index == -1:
				end_index = len(input)
			else:
				end_index = rparen_index
		elif rparen_index == -1:
			end_index = space_index
		else:
			end_index = min(space_index, rparen_index)
		return (SyntaxNode(input[position:end_index]), end_index)

def post_process(subtrees):
	# capitalize and add a period to each sub-sentence
	for child in subtrees:
		child.capitalize()
		if child.children[-1].label != ',' and child.children[-1].children[0].label not in {'?', '.'}:
			child.children.append(SyntaxNode('.', children=[SyntaxNode('.')]))
	return subtrees

def parse_stree(input):
	tree, _ = do_parse_stree(input, 0)
	return tree

def is_conjunction(stree_node):
	return stree_node.label == 'CC' and stree_node.children[0].label in {'and', 'but'}

def split_top_level_clauses(stree):
	# check if this is a conjunction
	if stree.label != 'S':
		return [stree]
	if stree.children[0].label != 'S':
		return [stree]

	s_children = [stree.children[0]]
	i = 1
	while i < len(stree.children):
		if stree.children[i].label == '.' and i + 1 == len(stree.children):
			break
		elif stree.children[i].label == ',':
			if i + 1 == len(stree.children):
				break
			else:
				i += 1
		elif is_conjunction(stree.children[i]):
			i += 1
		else:
			return [stree]
		if stree.children[i].label != 'S':
			return [stree]

		# recursively split any top-level clauses in this child node
		subclauses = split_top_level_clauses(stree.children[i])
		s_children.extend(subclauses)
		i += 1
	return s_children

def is_pp_attachment_phrase(stree):
	if stree.label != "S":
		return False
	if len(stree.children) < 3:
		return False
	child0 = stree.children[0]
	child1 = stree.children[1]
	child2 = stree.children[2]
	if child0.label != "S" or child1.label != "CC" or child2.label != "S":
		return False
	if len(child0.children) < 2 or len(child2.children) < 2:
		return False
	if child0.children[-1].label == "PP":
		return False
	if len(child0.children[-1].children) < 2 or len(child2.children[-1].children) < 2:
		return False
	if child0.children[-1].children[-1].label == "PP":
		return False
	if child2.children[-1].label != "PP" and child2.children[-1].children[-1].label != "PP":
		return False
	else:
		return True

def split_question(stree):
	if stree.label != 'SBARQ' and stree.label != 'S':
		return [stree]
	is_imperative = (stree.label == 'S')
	if stree.children[0].label != 'SBAR':
		return [stree]

	sbar = stree.children[0]
	if len(sbar.children) == 2 and sbar.children[0].label == 'IN' and len(sbar.children[0].children) == 1 and sbar.children[0].children[0].label.lower() == 'if' and sbar.children[1].label == 'S':
		s_children = split_top_level_clauses(sbar.children[1])
	else:
		return [stree]

	index = 1
	if stree.children[index].label == ',':
		index += 1
	if stree.children[index].label == 'ADVP' and len(stree.children[index].children) == 1 and stree.children[index].children[0].label == 'RB' and len(stree.children[index].children[0].children) == 1 and stree.children[index].children[0].children[0].label == 'then':
		index += 1
	if stree.children[index].label == 'RB' and len(stree.children[index].children) == 1 and stree.children[index].children[0].label == 'then':
		index += 1
	if is_imperative and stree.children[index].label == 'VP':
		# construct new imperative question
		s_children.append(SyntaxNode('S', stree.children[index:]))
		return s_children
	elif not is_imperative and stree.children[index].label in {'WHNP', 'WHADVP', 'WHADJP', 'WHPP', 'WHNP'} and stree.children[index + 1].label in {'SQ', 'S'}:
		# construct new question
		s_children.append(SyntaxNode('SBARQ', stree.children[index:]))
		return s_children
	else:
		return [stree]