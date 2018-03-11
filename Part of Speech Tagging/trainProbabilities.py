from __future__ import division
from collections import Counter

class TrainProbabilities:

	def __init__(self):

		self.probability_s = Counter()
		self.probability_s_given_s = Counter()
		self.probability_s2_given_s = Counter()
		self.probability_word_given_s = Counter()
		self.hmmsaved = Counter()
		self.VEsaved = Counter()
		self.total_s = 0
		self.total_s2 = 0
		self.total_words = 0
		self.total_sentences = 0

	def train(self, data):
		for sentences in data:
			words = sentences[0]
			part_of_speech = sentences[1]

			self.total_sentences+=1
			self.probability_s_given_s[part_of_speech[0]+'|'+'FW']+=1
			for i in range (2,len(part_of_speech)):
				self.probability_s2_given_s[part_of_speech[i]+'|'+part_of_speech[i-2]]+=1
				self.total_s2 +=1
			for i in range(len(words)):
				current_word = words[i]
				current_part_of_speech = part_of_speech[i]

				self.total_s+=1
				self.total_words+=1
				self.probability_s[current_part_of_speech]+=1
				self.probability_word_given_s[current_word+'|'+current_part_of_speech] +=1
				if i!=0:
					self.probability_s_given_s[current_part_of_speech+'|'+part_of_speech[i-1]]+=1
		all_part_of_speech = self.probability_s
		for current_part_of_speech in all_part_of_speech:
			self.probability_s_given_s[current_part_of_speech + '|'+'FW']+=1
			self.probability_s[current_part_of_speech]+=1
			self.total_s+=1
			for previous_part_of_speech in all_part_of_speech:
				self.probability_s_given_s[current_part_of_speech+'|'+previous_part_of_speech]+=1
				self.total_s+=2
				self.probability_s[previous_part_of_speech]+=1
				self.probability_s[current_part_of_speech]+=1
		for current_part_of_speech in all_part_of_speech:
			self.probability_s2_given_s[current_part_of_speech+'|'+'FW']+=1
			self.probability_s[current_part_of_speech]+=1
			self.total_s+=1
			for previous_part_of_speech in all_part_of_speech:
				self.probability_s2_given_s[current_part_of_speech+'|'+previous_part_of_speech]+=1
				self.total_s+=2
				self.probability_s[previous_part_of_speech]+=1
				self.probability_s[current_part_of_speech]+=1
		for i in self.probability_word_given_s:
			self.probability_word_given_s[i]/=self.probability_s[i.split("|")[1]]
		for i in self.probability_s_given_s:
			current_part_of_speech = i.split("|")
			if current_part_of_speech[1]=='FW':
				self.probability_s_given_s[i]/=self.total_sentences
			else:
				self.probability_s_given_s[i]/=self.probability_s[current_part_of_speech[0]]
		for i in self.probability_s:
			self.probability_s[i]/=self.total_s

	def get_all_part_of_speech(self):
		return self.probability_s
	def get_probability_w_given_s(self,word,s):
		return self.probability_word_given_s[word+'|'+s]
	def get_probability_s(self,s):
		return self.probability_s[s]
	def get_probability_s_given_s(self, previous_part_of_speech, current_part_of_speech):
		return self.probability_s_given_s[current_part_of_speech+'|'+previous_part_of_speech]
	def ifwordpresent(self,word):
		all_part_of_speech = self.probability_s
		for i in all_part_of_speech:
			if word+'|'+i in self.probability_word_given_s:
				return True
		return False
		for i in self.probability_word_given_s:
			word,part_of_speech =i.split("|")
			if i==word:
				return True
		return False
	def saveMarginalHMM(self, probab, part_of_speech, sentence):
		for i in range(len(sentence)):
			self.hmmsaved[sentence[i] + "|" + part_of_speech[i]] = probab[i]

	def saveMarginalVE(self, probab, part_of_speech, sentence):
		for i in range(len(sentence)):
			self.VEsaved[sentence[i] + "|" + part_of_speech[i]] = probab[i]
