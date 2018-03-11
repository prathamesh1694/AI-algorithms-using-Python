
from __future__ import division
import sys
from trainProbabilities import TrainProbabilities
import random
from random import randint
import math
from collections import Counter
import copy
import itertools
from operator import itemgetter

def find_next_word(word, exemplars):
    next_words = []
    for i in range(len(exemplars)):
        indices = [j for j, x in enumerate(exemplars[i][0]) if x == word]
        for k in indices:
            if k+1 != len(exemplars[i][0]): 
                next_words.append(exemplars[i][0][k+1])
    count2 = Counter(next_words).most_common(5)
    total = len(count2)-1
    if total!=-1:
        sample = randint(0,total)
    else:
        sample = 0
    return count2[sample][0]
def calculate_probability(sentence, confused_words_index, exemplars):
    prob = 1
    numerator =1
    denominator = 1
    for i in range(len(sentence)):
        if i==0 and i in confused_words_index:
            for j in range(len(exemplars)):
                if sentence[i] == exemplars[j][0][0]:
                    numerator+=1
            prob = numerator*1.0/len(exemplars)
        else:
            if i-1 in confused_words_index or i in confused_words_index:
                for j in range(len(exemplars)):
                    indices = [p for p, x in enumerate(exemplars[j][0]) if x == sentence[i-1]]
                    for k in indices:
                        denominator+=1
                        if k+1 !=len(exemplars[j][0]):
                            if sentence[i] == exemplars[j][0][k+1]:
                                numerator+=1
                if numerator!=0:
                    prob = prob * numerator*1.0/ denominator
    return prob


def do_part12(argv):
    train_data = argv[2]
    exemplars = []
    first_word = []
    five_sentences =[]
    file = open(train_data, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    for i in range(len(exemplars)):
        first_word.append(exemplars[i][0][0])
    count = Counter(first_word).most_common(6)
    for i in range(6):
        if i==1:
            continue
        flag =1
        sentences = ""
        next_word = count[i][0]
        sentences+= next_word
        while flag==1:
            if next_word != '.' and next_word!= '?' and next_word!= '!':
                next_word = find_next_word(next_word, exemplars)
                sentences+= ' '
                sentences+= next_word
            else:
                flag =0
        if i==0:
            print i+1,'. ',sentences,'\n'
        else:
            print i,'. ',sentences,'\n'


def do_part13(argv):
    input_sentence = argv[3].lower().split(" ")
    train_data = argv[2]
    exemplars = []
    confused_words = []
    probabilities =[]
    numerator = 1
    denominator = 1
    constant_prob =1
    confused_words_index = []
    confused_input_sentence = []
    set =[]
    file = open('confused_words5.txt', 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        confused_words += [ data ]
    file = open(train_data, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]
    for i in range(len(input_sentence)):
        for j in range(len(confused_words)):
            if input_sentence[i] in confused_words[j]:
                confused_input_sentence.append(i)
                confused_words_index.append(j)
    for i in confused_words_index:
        set.append(confused_words[i])
    permutations = list(itertools.product(*set))
    for i in range(len(input_sentence)):
        if i==0 and i not in confused_input_sentence:
            for j in range(len(exemplars)):
                if input_sentence[i] == exemplars[j][0][0]:
                    numerator+=1
            if numerator!=0:
                constant_prob = numerator*1.0/len(exemplars)
        else:
            if i-1 not in confused_input_sentence and i not in confused_input_sentence:
                for j in range(len(exemplars)):
                    indices = [p for p, x in enumerate(exemplars[j][0]) if x == input_sentence[i-1]]
                    for k in indices:
                        denominator+=1
                        if k+1 !=len(exemplars[j][0]):
                            if input_sentence[i] == exemplars[j][0][k+1]:
                                numerator+=1
                if numerator!=0:
                    constant_prob = constant_prob* numerator*1.0/ denominator

    for i in range(len(permutations)):
        temp = copy.deepcopy(input_sentence)
        for j in range(len(confused_input_sentence)):
            temp[confused_input_sentence[j]] = permutations[i][j]
        temp_prob = calculate_probability(temp, confused_input_sentence, exemplars)
        probabilities.append(temp_prob*constant_prob)
    max = 0
    ind =0
    maxindex = 0
    for i in range(len(probabilities)):
        if i==0:
            max = probabilities[i]
        if probabilities[i]>max:
            max=probabilities[i]
            maxindex=i
    if input_sentence == permutations[maxindex]:
        print "You do not need to change the sentence\n"
        print "It has the highest probability of ",probabilities[maxindex]
        print " ".join(input_sentence)
    else:
        print "The original sentence is "
        print " ".join(input_sentence)
        for i in range(len(permutations)):
            if input_sentence == permutations[i]:
                ind = i
        print "It has a probability ",probabilities[ind],'\n'
        print "The correct sentence would be"
        for j in range(len(permutations[maxindex])):
            input_sentence[confused_input_sentence[j]] = permutations[maxindex][j]
        print " ".join(input_sentence)
        print "It has a probability of ",probabilities[maxindex] 


    
        




class Solver:
    def __init__(self):
        self.probabilities = TrainProbabilities()

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label, o):
        posterior = 0.0
        for i in range(len(sentence)):
            if 'Simplified' in o:
                probab_w_given_s = self.probabilities.probability_word_given_s[sentence[i]+'|'+label[i]]
            elif 'HMM VE' in o:
                probab_w_given_s = self.probabilities.VEsaved[sentence[i]+'|'+label[i]]
            elif 'HMM MAP' in o:
                probab_w_given_s = self.probabilities.hmmsaved[sentence[i]+'|'+label[i]]
            else:
                return posterior
            probab_w_given_s = 0.01/self.probabilities.total_words if probab_w_given_s==0 else probab_w_given_s
            posterior+=math.log(probab_w_given_s)
        return posterior

    def train(self,data):
        self.probabilities.train(data)
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        predicted_part_of_speech = []
        probab_values =[]
        for i in sentence:
            part_of_speech =""
            highest_probab = 0

            for all_part_of_speech in self.probabilities.get_all_part_of_speech():
                probab_w_given_s = self.probabilities.get_probability_w_given_s(i,all_part_of_speech) * self.probabilities.get_probability_s(all_part_of_speech)    
                if probab_w_given_s > highest_probab:
                    highest_probab = probab_w_given_s
                    part_of_speech = all_part_of_speech
            predicted_part_of_speech.append(part_of_speech)
            probab_values.append(highest_probab)
        return predicted_part_of_speech

    def hmm_ve(self, sentence):
        all_part_of_speech = self.probabilities.get_all_part_of_speech()
        part_of_speech_index={}
        i=0
        for part_of_speech in all_part_of_speech:
            part_of_speech_index[part_of_speech] = i
            i+=1
        probab_max_values = [[(0,0)] * len(all_part_of_speech) for i in range(len(sentence))]
        first_column = self.get_all_part_of_speech_for_a_word(sentence[0],all_part_of_speech,self.probabilities)
        for i in first_column:
            first_column[i]*=self.probabilities.get_probability_s_given_s('FW',i)
        for i in first_column:
            probab_max_values[0][part_of_speech_index[i]] = (first_column[i], i)
        iteration = [i for i in probab_max_values[0] if i[0] > 0]
        VE = copy.deepcopy(iteration)
        iter =1
        for word in sentence[1:]:
            current_part_of_speech = self.get_all_part_of_speech_for_a_word(word, all_part_of_speech,self.probabilities)
            for part_of_speech in current_part_of_speech:
                temp =[]
                for previous_part__of_speech in iteration:
                    temp.append(tuple((previous_part__of_speech[0] * self.probabilities.get_probability_s_given_s(previous_part__of_speech[1],part_of_speech) * current_part_of_speech[part_of_speech], previous_part__of_speech[1])))
                if iter>1:
                    for previous_part__of_speech in VE:
                        temp.append(tuple((previous_part__of_speech[0] * self.probabilities.get_probability_s_given_s(previous_part__of_speech[1],part_of_speech) * current_part_of_speech[part_of_speech], previous_part__of_speech[1])))
                    VE = zip(current_part_of_speech.values(), current_part_of_speech.keys())
                if temp:
                    probab_max_values[iter][part_of_speech_index[part_of_speech]] = max(temp, key=itemgetter(0))
                else:
                    for i in probab_max_values[iter-1]:
                        if i[1]!=0:
                            Pos = (0.001, self.getk(part_of_speech_index, probab_max_values[iter-1].index(i)))
                            break
                    if iter>1:
                        for i in probab_max_values[iter -2]:
                            if i[1]!=0:
                                pos2 = (0.001, self.getk(part_of_speech_index, probab_max_values[iter-2].index(i)))
                                break
                    if self.probabilities.get_probability_s(Pos) > self.probabilities.get_probability_s(pos2):
                        probab_max_values[iter][part_of_speech_index[part_of_speech]] = (0.001, Pos)
                    else:
                        probab_max_values[iter][part_of_speech_index[part_of_speech]] = (0.001, pos2)
            iteration=[]
            for i in range(len(probab_max_values[iter])):
                if probab_max_values[iter][i][0]>0:
                    iteration.append((probab_max_values[iter][i][0], self.getk(part_of_speech_index,i)))
            iter+=1
        result =[]
        probab =[]
        for i in range(len(probab_max_values)):
            max_val = self.find_maximum(probab_max_values[i])
            result.append(probab_max_values[i][max_val][1])
            probab.append(probab_max_values[i][max_val][0])
        self.probabilities.saveMarginalVE(probab, result, sentence)
        return result



    def hmm_viterbi(self, sentence):
        all_part_of_speech = self.probabilities.get_all_part_of_speech()
        part_of_speech_index = {}
        i=0
        for part_of_speech in all_part_of_speech:
            part_of_speech_index[part_of_speech] = i
            i+=1
        probab_max_values = [[(0,0)] * len(all_part_of_speech) for i in range(len(sentence))]
        first_column = self.get_all_part_of_speech_for_a_word(sentence[0],all_part_of_speech, self.probabilities)
        for i in first_column:
            first_column[i]*=self.probabilities.get_probability_s_given_s('FW',i)
        for i in first_column:
            probab_max_values[0][part_of_speech_index[i]] = (first_column[i], i)
        iteration = [i for i in probab_max_values[0] if i[0] > 0]
        iter =1
        for word in sentence[1:]:
            current_part_of_speech = self.get_all_part_of_speech_for_a_word(word,all_part_of_speech,self.probabilities)
            for part_of_speech in current_part_of_speech:
                temp =[]
                temp = [(previous_part__of_speech[0] * self.probabilities.get_probability_s_given_s(previous_part__of_speech[1],part_of_speech)* current_part_of_speech[part_of_speech],previous_part__of_speech[1]) for previous_part__of_speech in iteration]
                if temp:
                    probab_max_values[iter][part_of_speech_index[part_of_speech]] = max(temp, key=itemgetter(0))
                else:
                    for i in probab_max_values[iter-1]:
                        if i[1]!=0:
                            probab_max_values[iter][part_of_speech_index[part_of_speech]] = (0.001, self.getk(part_of_speech_index,probab_max_values[iter-1].index(i)))
                            break
            iteration = []
            for i in range(len(probab_max_values[iter])):
                if probab_max_values[iter][i][0] >0:
                    iteration.append((probab_max_values[iter][i][0],self.getk(part_of_speech_index,i)))

            iter+=1
        result =[]
        probab= []
        l = len(probab_max_values)  
        last_column = probab_max_values[l-1]
        maximum_last_column = self.find_maximum(last_column)
        result.append(self.getk(part_of_speech_index,maximum_last_column))
        probab.append(probab_max_values[l-1][maximum_last_column][0])
        for i in range(l-1,0,-1):
            temp_val = probab_max_values[i][maximum_last_column]
            result.append(temp_val[1])
            probab.append(temp_val[0])
            maximum_last_column = part_of_speech_index[temp_val[1]]
        result.reverse()
        probab.reverse()
        self.probabilities.saveMarginalHMM(probab, result, sentence)
        return result
        

    def get_all_part_of_speech_for_a_word(self,word, all_part_of_speech, probabilities):
        all_possible_part_speech = {}
        if not probabilities.ifwordpresent(word):
            part_of_speech = 'noun'
            prob = 0.0001
            all_possible_part_speech[part_of_speech] = prob
        else:
            for part_of_speech in all_part_of_speech:
                probab = probabilities.get_probability_w_given_s(word, part_of_speech)
                if probab>0:
                    all_possible_part_speech[part_of_speech] =probab
        return all_possible_part_speech
    def getk(self,dict1, index):
        for key,value in dict1.iteritems():
            if value==index:
                return key
    def find_maximum(self,l):
        value =max(l,key=itemgetter(0))
        if value[1] == 0:
            for i in l:
                if i[1]!=0:
                    return l.index(i)
        return l.index(value)

 
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

