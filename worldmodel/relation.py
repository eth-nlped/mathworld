from worldmodel.container import Container
from worldmodel.tuple import EntityTuple
from worldmodel.quantity import Quantity
from worldmodel.misc import is_fraction, is_int, is_float

from sympy import Rational
import random

class Relation:

	def __init__(self, id, source, target):

		if not isinstance(id, str) and not isinstance(id, int):
			raise TypeError("id must be str or int")

		# note: we store source and target as containers rather than just container ids,
		# since we use the container objects to check for relation-specific constraints
		if not isinstance(source, Container):
			raise TypeError("source must be container")
		if not isinstance(target, Container):
			raise TypeError("target must be container")
		self.source = source
		self.target = target
		if isinstance(id, str):
			self.id = int(id) if is_int(id) else id
		else:
			self.id = id

	def __eq__(self, other):
		# id is not included
		# only structure for containers (rather than equality), for training data
		return isinstance(other, Relation) and self.source.equal_structure(other.source) \
			   and self.target.equal_structure(other.target)

	def __repr__(self):
		"""
		used to represent the complete object as a string, with argument names
		"""
		pass

	def __str__(self):
		"""
		used to represent the complete object as a string, without argument names
		used as output labels for the parser -- unknown quantities and other arguments will be None
		"""
		pass


	@property
	def source_id(self):
		return self.source.id

	@property
	def target_id(self):
		return self.target.id

	def _init_quantity(self, quantity):
		if not isinstance(quantity, float) and not isinstance(quantity, int) and \
				not isinstance(quantity, Rational) and not isinstance(quantity, str) and quantity is not None:
			raise TypeError("quantity must be float, int, str, Rational or None")
		if quantity is not None and not isinstance(quantity, str):
			# quantity known
			self.quantity = Quantity(name=None, num=quantity)
		elif quantity is None:
			# quantity not known
			# here we must generate a variable name, only needed if annotators have not done it
			self.quantity = Quantity(name=f"x{random.randint(5, 50)}")
		else:  # str
			quantity = quantity.strip()
			if is_fraction(quantity):
				# quantity known
				# fraction given as string
				self.quantity = Quantity(num=Rational(quantity))
			elif is_int(quantity):
				# quantity known
				self.quantity = Quantity(num=int(quantity))
			elif is_float(quantity):
				# quantity known
				self.quantity = Quantity(num=float(quantity))
			else:
				# quantity not known
				# assume str value is already the right variable name
				self.quantity = Quantity(name=quantity)
				# is there any good reason we're letting it be the input?
				# Yes, since if it is referred that way from the question container by the annotator

	def set_value(self, number):
		"""
		Sets the quantity to an explicit number
		"""
		self.quantity.set_value(number)

	def get_value(self):
		return self.quantity.get_value()

	def is_known(self):
		"""
		return True if quantity is explicit, False otherwise
		"""
		return self.quantity.is_known()

	def is_variable(self):
		"""
		return False if quantity is explicit, True otherwise
		"""
		return self.quantity.is_variable()

	def get_dict(self):
		pass

	@property
	def type(self):
		pass


class Transfer(Relation):

	def __init__(self, id, source, target, quantity, tuple, recipient = None, sender = None):
		super().__init__(id, source, target)

		# transfer-specific conditions
		if self.source.label != self.target.label:
			raise ValueError("container labels for source and target must match")
		if recipient is None and sender is None:
			raise ValueError("must have at least one of {recipient, sender}")
		if not isinstance(tuple, EntityTuple):
			raise TypeError("tuple must be entitytuple")
		if not isinstance(recipient, str) and recipient is not None:
			raise TypeError("recipient must be str or None")
		if not isinstance(sender, str) and sender is not None:
			raise TypeError("sender must be str or None")

		self._init_quantity(quantity)

		self.tuple = tuple
		self.recipient = recipient.lower() if recipient is not None else None
		self.sender = sender.lower() if sender is not None else None

		self.form = "transfer ( {} , {} , {} , {} , {} , {} )"

	def __eq__(self, other):
		return super().__eq__(other) and isinstance(other, Transfer) and self.quantity == other.quantity and \
			   self.tuple == other.tuple and self.recipient == other.recipient and self.sender == other.sender

	def equal_structure(self, other):
		return super().__eq__(other) and isinstance(other, Transfer) and \
			   self.tuple == other.tuple and self.recipient == other.recipient and self.sender == other.sender

	@property
	def type(self):
		return "transfer"

	def __repr__(self):
		return f"RelationTransfer(id={self.id},source={self.source.id},target={self.target.id},quantity={self.quantity.get_value()}," \
				   f"entity={self.tuple.entity},attribute={self.tuple.attribute},unit={self.tuple.unit},recipient={self.recipient},sender={self.sender})"

	def __str__(self):
		"""
		transfer(recipient_label, sender_label, quantity, entity, attribute, unit)
		"""
		# note: str of source and target label should match, and are thus interchangeable
		quantity = self.quantity if self.quantity.is_known() else None
		return self.form.format(self.recipient, self.sender, quantity, self.tuple.entity, self.tuple.attribute,
								self.tuple.unit)

	def get_dict(self):
		out = {}
		out["type"] = "transfer"
		out["id"] = self.id
		out["source"] = self.source
		out["target"] = self.target
		out["quantity"] = self.quantity.get_value()
		out["entity"] = self.tuple.entity
		out["attribute"] = self.tuple.attribute
		out["unit"] = self.tuple.unit
		out["recipient"] = self.recipient
		out["sender"] = self.sender
		return out

class Rate(Relation):

	def __init__(self, id, source, target, quantity, tuple_num, tuple_den):
		super().__init__(id, source, target)

		# rate-specific conditions
		#if source.tuple != tuple_num:
		#	print("Rate relation: Inconsistency with source container")
		#if target.tuple != tuple_den:
		#   print("Rate relation: Inconsistency with target container")
		if not isinstance(tuple_num, EntityTuple):
			raise TypeError("tuple_num must be entitytuple")
		if not isinstance(tuple_den, EntityTuple):
			raise TypeError("tuple_den must be entitytuple")

		self._init_quantity(quantity)
		self.tuple_num = tuple_num
		self.tuple_den = tuple_den

		self.form = "rate ( {} , {} , {} , {} , {} , {} , {} , {} )"

	def __eq__(self, other):
		return super().__eq__(other) and isinstance(other, Rate) and self.quantity == other.quantity and \
			   self.tuple_num == other.tuple_num and self.tuple_den == other.tuple_den

	def equal_structure(self, other):
		return super().__eq__(other) and isinstance(other, Rate) and \
			   self.tuple_num == other.tuple_num and self.tuple_den == other.tuple_den
	@property
	def type(self):
		return "rate"

	def __repr__(self):
		return f"RelationRate(id={self.id},source={self.source.id},target={self.target.id},quantity={self.quantity.get_value()},entity_num={self.tuple_num.entity}," \
				   f"attribute_num={self.tuple_num.attribute},unit_num={self.tuple_num.unit},entity_den={self.tuple_den.entity}," \
				   f"attribute_den={self.tuple_den.attribute},unit_den={self.tuple_den.unit})"

	def __str__(self):
		"""
		rate(container_label, quantity, num_entity, num_attr, num_unit, den_entity, den_attr, den_unit)
		"""
		quantity = self.quantity if self.quantity.is_known() else None
		return self.form.format(self.source.label, quantity, self.tuple_num.entity, self.tuple_num.attribute,
								self.tuple_num.unit, self.tuple_den.entity, self.tuple_den.attribute,
								self.tuple_den.unit)

	def get_dict(self):
		out = {}
		out["type"] = "rate"
		out["id"] = self.id
		out["source"] = self.source
		out["target"] = self.target
		out["quantity"] = self.quantity.get_value()
		out["entity_num"] = self.tuple_num.entity
		out["attribute_num"] = self.tuple_num.attribute
		out["unit_num"] = self.tuple_num.unit
		out["entity_den"] = self.tuple_den.entity
		out["attribute_den"] = self.tuple_den.attribute
		out["unit_den"] = self.tuple_den.unit
		return out


class PartWhole(Relation):

	def __init__(self, id, source, target):
		super().__init__(id, source, target)

		# set a placeholder quantity value
		self.quantity = Quantity(name="none")

		self.form = "part ( {} , {} , {} , {} , {} , {} , {} , {} )"

	def __eq__(self, other):
		return super().__eq__(other) and isinstance(other, PartWhole)

	def equal_structure(self, other):
		return self == other

	@property
	def type(self):
		return "part-whole"

	def set_value(self, number):
		pass

	def is_known(self):
		pass

	def get_part(self):
		return self.source

	def get_whole(self):
		return self.target

	def __repr__(self):
		return f"RelationPartWhole(id={self.id},source={self.source.id},target={self.target.id})"

	def __str__(self):
		"""
		part-whole(whole_label, part_label, whole_entity, whole_attr, whole_unit, part_entity, part_attr, part_unit)
		these are probably very difficult to parse -- doesn't seem to be the case that all this info
		occurs explicitly in the text
		"""
		return self.form.format(self.target.label, self.source.label, self.target.tuple.entity, self.target.tuple.attribute,
				   self.target.tuple.unit, self.source.tuple.entity, self.source.tuple.attribute, self.source.tuple.unit)

	def get_dict(self):
		out = {}
		out["type"] = "part-whole"
		out["id"] = self.id
		out["source"] = self.source
		out["target"] = self.target
		return out

class ExplicitAdd(Relation):

	def __init__(self, id, source, target, quantity, res_tuple, arg_tuple, result, argument):

		super().__init__(id, source, target)

		# explicit-specific conditions
		if source.label != argument.lower().strip() or target.label != result.lower().strip():
			raise ValueError("Explicit Add: Inconsistency in edge direction -- not from argument to result")
		if not isinstance(res_tuple, EntityTuple):
			raise TypeError("res_tuple must be entitytuple")
		if not isinstance(arg_tuple, EntityTuple):
			raise TypeError("arg_tuple must be entitytuple")
		if not isinstance(result, str):
			raise TypeError("result must be str")
		if not isinstance(argument, str):
			raise TypeError("argument must be str")

		self._init_quantity(quantity)
		self.res_tuple = res_tuple
		self.arg_tuple = arg_tuple
		self.result = result.lower().strip()
		self.argument = argument.lower().strip()

		self.form = "difference ( {} , {} , {} , {} , {} , {} , {} , {} , {} )"

	def __eq__(self, other):
		return super().__eq__(other) and isinstance(other, ExplicitAdd) and self.quantity == other.quantity and \
			   self.res_tuple == other.res_tuple and self.arg_tuple == other.arg_tuple \
			   and self.result == other.result and self.argument == other.argument

	def equal_structure(self, other):
		return super().__eq__(other) and isinstance(other, ExplicitAdd) and \
			   self.res_tuple == other.res_tuple and self.arg_tuple == other.arg_tuple and \
			   self.result == other.result and self.argument == other.argument

	@property
	def type(self):
		return "difference" #"explicit-add"

	def __repr__(self):
		return f"RelationExplicitAdd(id={self.id},source={self.source.id},target={self.target.id}," \
			   f"quantity={self.quantity.get_value()},res_entity={self.res_tuple.entity}," \
			   f"res_attribute={self.res_tuple.attribute},res_unit={self.res_tuple.unit}," \
			   f"arg_entity={self.arg_tuple.entity},arg_attribute={self.arg_tuple.attribute}," \
			   f"arg_unit={self.arg_tuple.unit},result={self.result},argument={self.argument})"

	def __str__(self):
		"""
		#explicit-add(result_label, argument_label, quantity, entity, attr, unit)
		explicit-add(result_label, argument_label, quantity, res_entity, res_attr, res_unit, arg_entity, arg_attr, arg_unit)
		"""
		quantity = self.quantity if self.quantity.is_known() else None
		return self.form.format(self.result, self.argument, quantity, self.res_tuple.entity, self.res_tuple.attribute,
								self.res_tuple.unit, self.arg_tuple.entity, self.arg_tuple.attribute, self.arg_tuple.unit)

	def get_dict(self):
		out = {}
		out["type"] = "difference" #"explicit-add"
		out["id"] = self.id
		out["source"] = self.source
		out["target"] = self.target
		out["quantity"] = self.quantity.get_value()
		out["res_entity"] = self.res_tuple.entity
		out["res_attribute"] = self.res_tuple.attribute
		out["res_unit"] = self.res_tuple.unit
		out["arg_entity"] = self.arg_tuple.entity
		out["arg_attribute"] = self.arg_tuple.attribute
		out["arg_unit"] = self.arg_tuple.unit
		out["result"] = self.result
		out["argument"] = self.argument
		return out

class ExplicitTimes(Relation):

	def __init__(self, id, source, target, quantity, res_tuple, arg_tuple, result, argument):

		super().__init__(id, source, target)

		# explicit-specific conditions
		if source.label != argument.lower() or target.label != result.lower():
			raise ValueError("Explicit Times: Inconsistency in edge direction -- not from argument to result")
		if not isinstance(res_tuple, EntityTuple):
			raise TypeError("res_tuple must be entitytuple")
		if not isinstance(arg_tuple, EntityTuple):
			raise TypeError("arg_tuple must be entitytuple")
		if not isinstance(result, str):
			raise TypeError("result must be str")
		if not isinstance(argument, str):
			raise TypeError("argument must be str")

		self._init_quantity(quantity)
		self.res_tuple = res_tuple
		self.arg_tuple = arg_tuple
		self.result = result.lower()
		self.argument = argument.lower()

		self.form = "explicit ( {} , {} , {} , {} , {} , {} , {} , {} , {} )"

	def __eq__(self, other):
		return super().__eq__(other) and isinstance(other, ExplicitTimes) and self.quantity == other.quantity and \
			   self.res_tuple == other.res_tuple and self.arg_tuple == other.arg_tuple \
			   and self.result == other.result and self.argument == other.argument

	def equal_structure(self, other):
		return super().__eq__(other) and isinstance(other, ExplicitTimes) and \
			   self.res_tuple == other.res_tuple and self.arg_tuple == other.arg_tuple and \
			   self.result == other.result and self.argument == other.argument

	@property
	def type(self):
		return "explicit" #"explicit-times"

	def __repr__(self):
		return f"RelationExplicitTimes(id={self.id},source={self.source.id},target={self.target.id}," \
			   f"quantity={self.quantity.get_value()},res_entity={self.res_tuple.entity}," \
			   f"res_attribute={self.res_tuple.attribute},res_unit={self.res_tuple.unit}," \
			   f"arg_entity={self.arg_tuple.entity},arg_attribute={self.arg_tuple.attribute}," \
			   f"arg_unit={self.arg_tuple.unit},result={self.result},argument={self.argument})"

	def __str__(self):
		"""
		explicit-times(result_label, argument_label, quantity, res_entity, res_attr, res_unit, arg_entity, arg_attr, arg_unit)
		"""
		quantity = self.quantity if self.quantity.is_known() else None
		return self.form.format(self.result, self.argument, quantity, self.res_tuple.entity, self.res_tuple.attribute,
								self.res_tuple.unit, self.arg_tuple.entity, self.arg_tuple.attribute, self.arg_tuple.unit)

	def get_dict(self):
		out = {}
		out["type"] = "explicit" #"explicit-times"
		out["id"] = self.id
		out["source"] = self.source
		out["target"] = self.target
		out["quantity"] = self.quantity.get_value()
		out["res_entity"] = self.res_tuple.entity
		out["res_attribute"] = self.res_tuple.attribute
		out["res_unit"] = self.res_tuple.unit
		out["arg_entity"] = self.arg_tuple.entity
		out["arg_attribute"] = self.arg_tuple.attribute
		out["arg_unit"] = self.arg_tuple.unit
		out["result"] = self.result
		out["argument"] = self.argument
		return out