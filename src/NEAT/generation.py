import random

from src.NEAT.nna import NeatNeuralNetwork
import copy
import sys
import math

def get_fit(elem):
	return elem.fitness


class Generation(object):
	def __init__(self, population, data, label):
		self.data = data / 255
		self.label = label
		self.population = []
		for i in range(population):
			nna = NeatNeuralNetwork(len(data[0]), 10, 20)
			nna.mutate()
			self.population.append(nna)

	def cleanFitness(self):
		for elem in self.population:
			elem.fitness = 0

	def run_generation(self):
		lendata = len(self.data)
		lenpopu = len(self.population)
		perc = 0
		for i in range(lenpopu):
			for y in range(lendata):
				out = self.population[i].run(self.data[y])
				for z in range(len(out)):
					if z == self.label[y]:
						self.population[i].fitness += out[z] * 10
					else:
						self.population[i].fitness += 1 - out[z]
			#if i / 10 > perc:
			sys.stdout.write('#')
			sys.stdout.flush()
			perc += 1

		self.population.sort(key=get_fit, reverse=True)
		print("")
		print("Accuracy: ", (self.population[0].fitness / (lendata * 19.0)) * 100.0, "%")
		print("max: ", self.population[0].fitness)
		print("min: ", self.population[-1].fitness)
		print("")

	def reproduce(self):
		for i in range(10, len(self.population)):
			mother = self.population[random.randint(0, 9)]
			self.population[i] = copy.deepcopy(mother)
			child = self.population[i]
			child.mutate()
			child.clean()


from src.helper import get_label, pre_processed_data, pre_processed_label
from src.arg import parse_args
import numpy as np
import json

if __name__ == '__main__':
	args = parse_args("NEAT").parse_args(["-r", "-s", "0.01", "-f", "../../data/random/"])
	rand = np.random.randint(0, 10000000)
	data, testdata = pre_processed_data(args, rand)
	label, testlabel = pre_processed_label(args, rand)
	gen = Generation(100, data, label)

	maxFit = 0
	for i in range(100):
		print("Running generation ", i, ":")
		gen.cleanFitness()
		gen.run_generation()
		gen.reproduce()
		if maxFit < gen.population[0].fitness:
			gen.population[0].save("best_" + str(math.floor((gen.population[0].fitness / (len(data) * 19.0)) * 100.0)) + ".txt")
			maxFit = gen.population[0].fitness
