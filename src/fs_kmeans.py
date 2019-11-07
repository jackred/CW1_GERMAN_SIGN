from helper import get_data_value, get_label, pre_processed_data, pre_processed_label
from arg import parse_args
import random
import sys
import numpy as np


class Cluster:
	def __init__(self, size):
		self.dimensions = [0.0] * size
		self.size = size
		self.nb_train = 0

	def clean_dimensions(self):
		self.nb_train = 0
		for i in range(len(self.dimensions)):
			self.dimensions[i] = 0.0

	def add_training_set(self, training):
		if len(self.dimensions) != len(training):
			print("Wrong size !")
			return
		self.nb_train += 1
		for i in range(self.size):
			self.dimensions[i] += training[i]

	def calculate_K(self):
		for i in range(self.size):
			if self.nb_train != 0:
				self.dimensions[i] = self.dimensions[i] / self.nb_train


def save_image(data, j):
	with open('../test' + str(j) + '.ppm', 'w+') as f:
		f.write('P2\n48 48 255\n')
		strdata = [str(int(i)) for i in data]
		f.write('\n'.join(strdata))


class KMEANS:
	def __init__(self, data, nbclass, labels=[]):
		self.nbclass = nbclass
		self.data = data
		self.data_affiliation = []
		self.clusters = []
		for i in range(len(data)):
			self.data_affiliation.append(random.randint(0, nbclass - 1))
		for i in range(nbclass):
			self.clusters.append(Cluster(len(self.data[0])))
		if len(labels) == len(data):
			for i in range(len(labels)):
				self.clusters[labels[i]].add_training_set(data[i])
			for i in range(nbclass):
				self.clusters[i].calculate_K()
				save_image(self.clusters[i].dimensions, i)

	def E(self):
		for i in range(len(self.clusters)):
			self.clusters[i].clean_dimensions()
		for i in range(len(self.data_affiliation)):
			self.clusters[self.data_affiliation[i]].add_training_set(self.data[i])
		for cluster in self.clusters:
			cluster.calculate_K()

	def M(self):
		#print(self.data_affiliation)
		prcent = 0
		""" For every image """
		for i in range(len(self.data)):
			if i / 100 > prcent:
				sys.stdout.write('#')
				sys.stdout.flush()
				prcent += 1
			tmp = [0] * self.nbclass
			""" For every sign """
			for y in range(self.nbclass):
				""" For every pixel of the image """
				for z in range(len(self.data[0])):
					""" class += (pval(y, z) - val(i, z) """
					tmp[y] += (self.clusters[y].dimensions[z] - self.data[i][z]) ** 2
			max = tmp[0]
			id = 0
			for y in range(self.nbclass):
				if tmp[y] < max:
					id = y
			self.data_affiliation[i] = id
		print("")

	def find_nearest_cluster(self, img):
		tmp = [0] * self.nbclass
		""" For every sign """
		for y in range(self.nbclass):
			""" For every pixel of the image """
			for z in range(len(self.data[0])):
				""" class += (pval(y, z) - val(i, z) """
				tmp[y] += (self.clusters[y].dimensions[z] - img[z]) ** 2
		max = tmp[0]
		id = 0
		for y in range(self.nbclass):
			if tmp[y] < max:
				id = y
		return id


if __name__ == '__main__':
	args = parse_args("kmeans").parse_args(["-r", "-s", "0.01"])
	rand = np.random.randint(0, 10000000)
	print("Fetching data:")
	data, testdata = pre_processed_data(args, rand)
	print("Done")
	print("Fetching labels:")
	label, testlabel = pre_processed_label(args, rand)
	print("Done")
	kmeans = KMEANS(data, 10, label)

	print("Main Loop:")
	for i in range(10):
		print(i)
		print("E:")
		kmeans.E()
		print("Done")
		print("M:")
		kmeans.M()
		print("Done\n")
		for i in range(len(kmeans.clusters)):
			save_image(kmeans.clusters[i].dimensions, i)

		tmp = [0] * 10
		for elem in kmeans.data_affiliation:
			tmp[elem] += 1
		print(tmp)

	total = 0
	right = 0
	for i in range(len(testdata)):
		total += 1
		if testlabel[i] == kmeans.find_nearest_cluster(testdata[i]):
			right += 1
		print("Accuracy : ", round((right / total) * 100, 3), " %")


