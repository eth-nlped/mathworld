
class EntityTuple:

	def __init__(self, entity, attribute=None, unit=None):

		if not isinstance(entity, str):
			raise TypeError("TypeError: entity must be str")
		if not isinstance(attribute, str) and attribute is not None:
			raise TypeError("TypeError: attribute must be str or None")
		if not isinstance(unit, str) and unit is not None:
			raise TypeError("TypeError: unit must be str or None")

		self.entity = entity
		self.attribute = attribute
		self.unit = unit

	def __eq__(self, other):
		return isinstance(other, EntityTuple) and self.entity == other.entity and \
			   self.attribute == other.attribute and self.unit == other.unit

	def __repr__(self):
		return f"EntityTuple(entity={self.entity},attribute={self.attribute},unit={self.unit})"

	def get_tuple(self):
		return (self.entity, self.attribute, self.unit)

	def get_dict(self):
		return {"entity": self.entity,
				"attribute": self.attribute,
				"unit": self.unit
				}

