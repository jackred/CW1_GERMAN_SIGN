from math import exp


def sigmoid(x):
	return 1 / (1 + exp(-x))


class Node(object):

	def __init__(self):
		self.total = 0.0
		self.activated = 0.0

		self.prev_link = []
		self.next_link = []

	def activate(self):
		if self.prev_link.__len__() != 0:
			self.total = 0
		for connection in self.prev_link:
			self.total += connection.prev.activated * connection.weight
		self.activated = sigmoid(self.total)
