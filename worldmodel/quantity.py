from sympy import symbols
from sympy import Rational

class Quantity:

	def __init__(self, name = None, num = None):
		if not isinstance(name, str) and name is not None:
			raise TypeError("name must be str or None")
		if not isinstance(num, float) and not isinstance(num, int) and not isinstance(num, Rational) \
				and num is not None:
			raise TypeError("num must be int, float, Rational or None")
		if name is None and num is None:
			raise ValueError("both name and num cannot be None")
		self.var = symbols(name) if name is not None else None
		self.num = num

	def __eq__(self, other):
		return isinstance(other, Quantity) and self.var == other.var and self.num == other.num

	def __repr__(self):
		return f"Quantity(number={self.num},variable={self.var})"

	def __str__(self):
		if self.is_known():
			return str(self.num)
		else:
			return str(self.var)

	def is_variable(self):
		if self.num is None:
			return True
		else:
			return False

	def is_known(self):
		return not self.is_variable()

	def set_value(self, num):
		if not isinstance(num, float) and not isinstance(num, int):
			raise TypeError("num must be int or float")
		if self.is_known():
			raise ValueError("Quantity already has value")
		else:
			self.num = num

	def get_value(self):
		"""
		return variable if unknown, num otherwise
		"""
		if self.is_known():
			return self.num
		else:
			return self.var
