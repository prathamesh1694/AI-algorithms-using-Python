import sys
import math
import heapq
import numpy as np
import time

class Image:
    def __init__(self, row):
        d = row.split()
        self.id = d[0]
        self.orientation = int(d[1])
        self.data = []
        i = 2
        while i < len(d):
            self.data.append(int(d[i]))
            i = i + 1
def get_orientation(x):
    if x==0:
        return 0
    elif x==1:
        return 90
    elif x==2:
        return 180
    elif x==3:
        return 270

def neural_network_train(input_file,output_file,start_time):
    X1=[]
    D1 = []
    #count = 25000
    print "loading data"
    training = loadData(input_file)
    for i in training:
        D1.append(get_output_class(i.orientation))
        inp = []
        for k in range(0,192):
            z = i.data[k]
            z = z/255
            inp.append(z)
        inp.append(1)
        X1.append(inp)
        #count-=1
        #if count == 0:
        #   break
    X = np.array([np.array(x1) for x1 in X1])
    D = np.array([np.array(d1) for d1 in D1])
    
    w_ij = np.random.uniform(-1,1,(130,193))
    w_jk = np.random.uniform(-1,1,(4,131))

    delta_w_ij = np.zeros((130,193))
    delta_w_jk = np.zeros((4,131))

    epoch = 0
    alpha = 0.2
    eta = 0.4
    print "BackPropogation"
    while True:
        diff = []
        for i in range(X.shape[0]):
            x = np.reshape(X[i,],(193,1))
            d = np.reshape(D[i],(4,1))

            v_j = w_ij.dot(x)
            y_j = activation_function(v_j)
            y_j = np.append(y_j,[[1]], axis = 0)
            v_k = w_jk.dot(y_j)
            y_k = activation_function(v_k)
            e_k = d - y_k
            
            delta_k = e_k * y_k * (1-y_k)
            delta_j = w_jk.T.dot(delta_k)*y_j*(1-y_j)
            
            delta_w_jk = eta*delta_k.dot(y_j.T) + alpha*delta_w_jk
            delta_w_ij = eta*delta_j[:-1].dot(x.T) + alpha*delta_w_ij
            w_ij+= delta_w_ij
            w_jk+= delta_w_jk

            diff.append(np.sum(e_k))
        epoch+=1
        error = np.max(np.abs(diff))
        print 'epoch:',epoch,'\nerror:',error,'\n'
        if error<0.30:
            break
        if epoch == 300:
            break
    np.savez(output_file,w_ij=w_ij,w_jk=w_jk)
    if epoch==300:
        print "Maximum epochs speficied reached"
        print "Please test with \' python orient.py nnet test <test-data file name> <second filename used in training command>"
    print("--- %s seconds ---" % (time.time() - start_time))

def neural_network_test(input_file, weights):
    filename = weights+ ".npz"
    data = np.load(filename)
    w_ij = data['w_ij']
    w_jk = data['w_jk']
    correct = 0 
    total = 0
    confusion_matrix = np.zeros((4,4))  
    testing = loadData(first_file)
    for i in testing:
        total+=1
        X1 = []
        final = i.orientation
        inp = []
        for k in range(0,192):
            z = i.data[k]
            z=z/255
            inp.append(z)
        inp.append(1)
        X1.append(inp)
        X = np.array([np.array(x1) for x1 in X1])
        x = np.reshape(X1,(193,1))
        v_j = w_ij.dot(x)
        y_j = activation_function(v_j)
        y_j = np.append(y_j,[[1]], axis = 0)
        v_k = w_jk.dot(y_j)
        y_k = activation_function(v_k)
        #outp = y_k.index(np.max(y_k))
        calculated_orientation = get_orientation(np.where(y_k==np.max(y_k))[0][0])
        if i.orientation == calculated_orientation:
            correct+=1
        class_accu = correct/total * 100
        if i.orientation ==0:
            if calculated_orientation == 0:
                confusion_matrix[0][0]+=1
            elif calculated_orientation == 90:
                confusion_matrix[0][1] +=1
            elif calculated_orientation == 180:
                confusion_matrix[0][2] +=1
            elif calculated_orientation == 270:
                confusion_matrix[0][3]+=1
        elif i.orientation ==90:
            if calculated_orientation == 0:
                confusion_matrix[1][0]+=1
            elif calculated_orientation == 90:
                confusion_matrix[1][1] +=1
            elif calculated_orientation == 180:
                confusion_matrix[1][2] +=1
            elif calculated_orientation == 270:
                confusion_matrix[1][3]+=1
        elif i.orientation ==180:
            if calculated_orientation == 0:
                confusion_matrix[2][0]+=1
            elif calculated_orientation == 90:
                confusion_matrix[2][1] +=1
            elif calculated_orientation == 180:
                confusion_matrix[2][2] +=1
            elif calculated_orientation == 270:
                confusion_matrix[2][3]+=1
        elif i.orientation ==270:
            if calculated_orientation == 0:
                confusion_matrix[3][0]+=1
            elif calculated_orientation == 90:
                confusion_matrix[3][1] +=1
            elif calculated_orientation == 180:
                confusion_matrix[3][2] +=1
            elif calculated_orientation == 270:
                confusion_matrix[3][3]+=1
    matrix = confusion_matrix
    file = open("nnet_output.txt",'w')
    file.write("Classified As =>         0          90          180          270\n")
    file.write("Actual Value          \n")
    file.write("      \\/\n")
    file.write("        0              " + str(matrix[0][0]) + "          " + str(matrix[0][1]) + "          " + str(matrix[0][2]) + "           " + str(matrix[0][3])+"\n")
    file.write("       90              " + str(matrix[1][0]) + "          " + str(matrix[1][1]) + "          " + str(matrix[1][2]) + "           " + str(matrix[1][3])+"\n")
    file.write("      180              " + str(matrix[2][0]) + "          " + str(matrix[2][1]) + "          " + str(matrix[2][2]) + "           " + str(matrix[2][3])+"\n")
    file.write("      270              " + str(matrix[3][0]) + "          " + str(matrix[3][1]) + "          " + str(matrix[3][2]) + "           " + str(matrix[3][3])+"\n")
    file.write(" \n")
    file.write("The classification accuracy is " + str(class_accu) +"%\n")
    file.close()
    print "Completed"

def get_output_class(input):
    if input == 0:
        return [1,0,0,0]
    elif input == 90:
        return [0,1,0,0]
    elif input == 180:
        return [0,0,1,0]
    elif input == 270:
        return [0,0,0,1]

def activation_function(v):                                                 #sigmoid func
    return 1/(1+np.exp(-v))

def distanceFromImage(img1, img2):
    result = 0
    i = 0
    while(i < len(img1.data)):
        result = result + math.pow(img1.data[i] - img2.data[i], 2)
        i = i + 1
    return math.sqrt(result)

def loadData(file):
    images = []
    f = open(file)
    for row in f:
        images.append(Image(row))
    return images

def printMatrix(matrix):
    print "Classified As =>         0          90          180          270"
    print "Actual Value          "
    print "      \\/"
    print "        0              " + str(matrix[0][0]) + "          " + str(matrix[0][1]) + "          " + str(matrix[0][2]) + "           " + str(matrix[0][3])
    print "       90              " + str(matrix[1][0]) + "          " + str(matrix[1][1]) + "          " + str(matrix[1][2]) + "           " + str(matrix[1][3])
    print "      180              " + str(matrix[2][0]) + "          " + str(matrix[2][1]) + "          " + str(matrix[2][2]) + "           " + str(matrix[2][3])
    print "      270              " + str(matrix[3][0]) + "          " + str(matrix[3][1]) + "          " + str(matrix[3][2]) + "           " + str(matrix[3][3])
    print " "

if len(sys.argv) < 4:
    print "Usage: one of "
    print "    ./orient.py nearest train_file.txt test_file.txt"
    print "    ./orient.py nnet train_file.txt model_file.txt"
    print "    ./orient.py nnet test_file.txt model_file.txt"
    sys.exit()

algorithm = sys.argv[1]


if algorithm == "nearest":
    first_file = sys.argv[2]
    second_file = sys.argv[3]
    training = loadData(first_file)
    f = open(second_file)
    testing = []
    count = 0.0
    correct = 0.0

    matrix = [[0 for x in range(4)] for y in range(4)]
    output_file = open('nearest_output.txt', 'w')
    for row in f:
        t = Image(row)
        #t.distances = []
        nearest = None
        min_distance = None
        testing.append(t)
        count = count + 1

        for tr in training:
            distance = distanceFromImage(t, tr)
            if min_distance == None or min_distance > distance:
                min_distance = distance
                nearest = tr
        # heapq.heappush(t.distances, (distanceFromImage(t, tr), tr))
        matrix[t.orientation/90][nearest.orientation/90] += 1
        if t.orientation == nearest.orientation:
            correct = correct + 1
        output_file.write(t.id + ' ' + str(nearest.orientation) + '\n')
    # print heapq.heappop(testing[0].distances)[1].orientation
    output_file.close()
    print '\nClassification Accuracy: ' + str(round(((correct / count) * 100.00), 2)) + '%\n'
    printMatrix(matrix)


elif algorithm == "nnet":
    start_time = time.time()
    if sys.argv[2] == "train":
        first_file = sys.argv[3]
        second_file = sys.argv[4]
        neural_network_train(first_file, second_file,start_time)
    elif sys.argv[2] == "test":
        first_file = sys.argv[3]
        second_file = sys.argv[4]
        neural_network_test(first_file, second_file)
    else:
        print "Please use second parameter as test or train"
else:
    print "Invalid algorithm"
    sys.exit()