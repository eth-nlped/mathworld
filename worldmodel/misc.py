import re

def is_fraction(string):
	"""
	return true if string is a fraction
	"""
	if re.fullmatch("[0-9]+/[0-9]+", string):
		return True
	else:
		return False

def is_int(string):
	"""
	return true if string is an integer
	"""
	if re.fullmatch("[0-9]+", string):
		return True
	else:
		return False

def is_float(string):
	"""
	return true if string is a float
	"""
	if re.fullmatch("[0-9]*\.?[0-9]*", string):
		return True
	else:
		return False
