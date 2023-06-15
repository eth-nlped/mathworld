from worldmodel.tuple import EntityTuple
from worldmodel.quantity import Quantity
from worldmodel.misc import is_fraction, is_int, is_float

import sympy
from sympy import symbols, Rational
import random

class Container:

	def __init__(self, id, label, entity, quantity = None, attribute = None, unit = None, tuple = None):
		if not isinstance(id, str) and not isinstance(id, int):
			raise TypeError("id must be str or int")
		if not isinstance(label, str) and label is not None:
			raise TypeError("label must be str")
		if not isinstance(entity, str):
			raise TypeError("entity must be str")
		if not isinstance(quantity, float) and not isinstance(quantity, int) and \
				not isinstance(quantity, Rational) and not isinstance(quantity, str) and quantity is not None:
			raise TypeError("quantity must be float, int, str, Rational or None")
		if not isinstance(attribute, str) and attribute is not None:
			raise TypeError("attribute must be str or None")
		if not isinstance(unit, str) and unit is not None:
			raise TypeError("unit must be str or None")

		if isinstance(id, str):
			self.id = int(id) if is_int(id) else id
		else:
			self.id = id
		if label is None:
			self.label = "world"
		else:
			self.label = label.lower().strip()

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

		if isinstance(tuple, EntityTuple):
			self.tuple = tuple
		else:
			self.tuple = EntityTuple(entity.strip(), attribute if attribute is None else attribute.strip(),
									 unit if unit is None else unit.strip())

		self.form = "container ( {} , {} , {} , {} , {} )"


	def __eq__(self, other):
		# include id?
		return isinstance(other, Container) and self.label == other.label and self.quantity == other.quantity and \
			   self.tuple == other.tuple

	def __repr__(self):
		"""
		used to represent the complete object as a string
		"""
		return f"Container(id={self.id},label={self.label},quantity={self.quantity.get_value()},entity={self.tuple.entity}," \
				   f"attribute={self.tuple.attribute},unit={self.tuple.unit})"

	def __str__(self):
		"""
		used as data representation for parser
		"""
		quantity = self.quantity if self.quantity.is_known() else None
		return self.form.format(self.label, quantity, self.tuple.entity, self.tuple.attribute, self.tuple.unit)

	def get_dict(self):
		out = {}
		out["id"] = self.id
		out["label"] = self.label
		out["quantity"] = self.quantity.get_value()
		out["entity"] = self.tuple.entity
		out["attribute"] = self.tuple.attribute
		out["unit"] = self.tuple.unit
		return out

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

	def equal_structure(self, other, soft=False):
		"""
		return true of self has same structure as other
		structure equal iff label and entity tuple both match
		soft macthes label and entity inly
		"""
		if not isinstance(other, Container):
			return False
		if not soft:
			return self.label == other.label and self.tuple == other.tuple
		else:
			return self.label == other.label and self.tuple.entity == other.tuple.entity
