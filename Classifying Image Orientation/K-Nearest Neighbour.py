import numpy as np

class Dot(object):
    def __init__(self,rgb):
        self.rgb = rgb

    def dis(self,dot2):
         return np.linalg.norm(np.array(self.rgb) - np.array(dot2.rgb))

class Image(object):
    def __init__(self,name,label):
        self.name = name
        self.label = label
        self.color = []

    def color_format(self,rgb):
        self.color = [Dot(rgb[i:i+3]) for i in range(0,len(rgb),3)]
        return

    def distance(self,image2):
        d = 0
        for dot1, dot2 in zip(self.color,image2.color):
            d += dot1.dis(dot2)
        return d

class Pool(object):
    def __init__(self):
        self.test = []
        self.training = []

    def read_data(self, file, learn = False):
        doc = open(file,'r')
        for line in doc:
            word = line.split(' ')
            image = Image(word[0],word[1])
            image.color_format(tuple(map(int,word[2:])))
            if learn:
                self.training.append(image)
            else: self.test.append(image)
        print('Finished reading %s'%file)
        return

    def nearest(self):
        min = 100000
        with open('nearest_output.txt', 'a', encoding='latin-1') as f:
            for image in self.test:
                print('calculating...')
                for imag in self.training:
                    d = image.distance(imag)
                    if d < min:
                        min = d
                        min_name = (imag.name, image.label)
                f.write(str(min_name[0]) + ' ' + str(min_name[1]) + '\n')
        print('Finished writing nearest_output.txt')
        return


    def get_training(self):
        return self.training

p = Pool()
p.read_data('train-data.txt',learn = True)
p.read_data('test-data.txt')
p.nearest()
