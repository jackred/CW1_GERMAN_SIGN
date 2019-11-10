from random import uniform
from random import randint

import math
from src.NEAT.node import Node
from src.NEAT.connection import Connection


class NeatNeuralNetwork(object):
	def __init__(self, input, output, brain_cycle):
		self.input_size = input
		self.output_size = output
		self.brain_cycle = brain_cycle
		self.fitness = 0
		self.input = []
		self.output = []
		self.neurons = []
		self.dna = []
		self.compute_node = []

		# Setting Nodes
		for i in range(0, input):
			node = Node()
			self.input.append(node)
			self.neurons.append(node)

		for i in range(0, output):
			node = Node()
			self.output.append(node)
			self.neurons.append(node)

		self.mutate()

	def save(self, path):
		f = open(path, "w+")
		f.write(str(self.neurons.__len__()) + '\n')
		f.write(str(self.dna.__len__()) + '\n')
		for elem in self.dna:
			f.write(str(self.neurons.index(elem.prev)) + '\n')
			f.write(str(elem.weight) + '\n')
			f.write(str(self.neurons.index(elem.next)) + '\n')

	def load(self, path):
		f = open(path, "r")
		lenneurons = int(f.readline())
		lenconnections = int(f.readline())
		for i in range(self.neurons.__len__(), lenneurons):
			self.neurons.append(Node())
		for i in range(lenconnections):
			prev = int(f.readline())
			weight = float(f.readline())
			next = int(f.readline())
			self.dna.append(Connection(self.neurons[prev], self.neurons[next]))
			self.dna[-1].weight = weight

	def clean(self):
		for node in self.neurons:
			node.total = 0
			node.activated = 0

	def run(self, input_data):
		for i in range(0, len(input_data)):
			if input_data[i] == 1.0:
				input_data[i] = 0.999999
			self.input[i].total = math.log10(input_data[i]/(1-input_data[i])) if input_data[i] > 0 else -math.log10(-input_data[i] / (1-(-input_data[i])))
			self.compute_node.append(self.input[i])

		for i in range(self.brain_cycle):
			tmp = []
			for node in self.compute_node:
				node.activate()
				for next_connec in node.next_link:
					if not next_connec.next in tmp:
						tmp.append(next_connec.next)
			self.compute_node = tmp

		out = []
		for neuron in self.output:
			out.append(neuron.activated)
		return out

	def mutate_connection(self):
		_from = self.neurons[randint(0, self.neurons.__len__() - 1)]
		while _from in self.output or _from.next_link.__len__() == self.neurons.__len__() - self.input_size:
			_from = self.neurons[randint(0, self.neurons.__len__() - 1)]

		tmp = []
		for elem in _from.next_link:
			tmp.append(elem.next)

		_to = self.neurons[randint(0, self.neurons.__len__() - 1)]
		while _to == _from or _to in tmp or _to in self.input:
			_to = self.neurons[randint(0, self.neurons.__len__() - 1)]

		connection = Connection(_from, _to)
		connection.weight = uniform(-1.0, 1.0)
		_from.next_link.append(connection)
		_to.prev_link.append(connection)
		self.dna.append(connection)

	def mutate_node(self):
		#return
		exi_connection = self.dna[randint(0, self.dna.__len__() - 1)]
		# return
		new_node = Node()
		new_connection = Connection(new_node, exi_connection.next)
		new_connection.weight = 1.0

		new_connection.prev = new_node
		new_connection.next = exi_connection.next
		new_connection.next.prev_link.append(new_connection)

		exi_connection.next = new_node

		new_node.prev_link.append(exi_connection)
		new_node.next_link.append(new_connection)

		new_connection.next.prev_link.remove(exi_connection)
		self.dna.append(new_connection)
		self.neurons.append(new_node)

	def mutate_weight(self):
		if uniform(0.0, 1.0) < 0.3:
			self.dna[randint(0, self.dna.__len__() - 1)].weight += uniform(-0.2, 0.2)
		else:
			self.dna[randint(0, self.dna.__len__() - 1)].weight = uniform(-1.0, 1.0)

	def mutate(self):
		rd = uniform(0.0, 1.0)
		if rd < 0.3:
			for i in range(1):
				if uniform(0.0, 1.0) > 0.2:
					self.mutate_connection()
		elif rd < 0.4:
			for i in range(1):
				if self.dna.__len__() != 0 and uniform(0.0, 1.0) > 0.5:
					self.mutate_node()
		else:
			for i in range(1):
				if self.dna.__len__() != 0 and uniform(0.0, 1.0) > 0.3:
					self.mutate_weight()


