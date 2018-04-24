# Character-level language model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Basic I/O
data = open("Simple_text.txt",'r').read()	  #read from an input file
vocab_size = len(set(data))				
chars = set(data)

chartoix = {ch:ix for ix,ch in enumerate(chars)}		#mapping of character to indices
ixtochar = {ix:ch for ix,ch in enumerate(chars)}		#mapping of indices to character


# Hyperparameters
n_x = n_y = vocab_size # Input and output size
n_a = 100	#Hidden units
alpha = 0.001 	#Learning Rate
T = 50 	#Sequence Length

# This RNN is a simple neural network with one hidden layer in a loop
Wxa = np.random.randn(n_a,n_x)*0.01
Waa = np.random.randn(n_a,n_a)*0.01
b = np.zeros((n_a,1))
Way = np.random.randn(n_y,n_a)*0.01
by = np.zeros((n_y,1))

#Inputs, hidden activation units and outputs
x,a,y,p,target1 = {},{},{},{},{}

#Softmax function
def softmax(x):
    #Compute softmax values for each sets of scores in x.
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def loss(a_prev,target,index):
	# global Wxa,Way,Waa,by,b
	cross_entropy_loss = 0
	a[-1] = a_prev
	#Forward propagation in time
	for t in range(T):
		#Input
		x[t] = np.zeros([vocab_size,1])
		x[t][index[t]] = 1 	#one hot vector for input X

		target1[t] = target[t].reshape(n_y,1)

		a[t] = np.tanh(np.dot(Wxa,x[t]) + np.dot(Waa,a[t-1]) + b) 	#hidden layer output
		y[t] = np.dot(Way,a[t])+by 	#Output vector y
		p[t] = softmax(y[t]) 	#Softmax Probabilities
		# print (ixtochar[np.argmax(x[t])],ixtochar[np.argmax(p[t])])
		# print ([np.argmax(x[t])]," ",[np.argmax(p[t])])
		
		
		cross_entropy_loss +=  -np.sum(np.multiply(target1[t],np.log(p[t]))) 	#Cross Entropy loss
		# cross_entropy_loss +=  -np.sum(np.multiply(target1[t],np.log(p[t]))) - np.sum(np.multiply(1-target1[t],np.log(1-p[t]))) 	#Cross Entropy loss
		

	#Backpropagation in time
	dWxa, dWaa, dWay = np.zeros_like(Wxa), np.zeros_like(Waa), np.zeros_like(Way)
	db, dby = np.zeros_like(b), np.zeros_like(by)
	for t in reversed(range(T)):
		dy = p[t] - target1[t]
		dWay += np.dot(dy,(a[t]).transpose())
		dby += dy
		da = np.dot(Way.transpose(),dy)
		dz = np.multiply(da,1-np.power(a[t],2))
		dWaa += np.dot(dz,a[t-1].transpose())
		dWxa += np.dot(dz,x[t].transpose())
		db += dz

	#Gradient Clipping
	for gradients in [dWay, dby, dWaa, dWxa, db]:
		np.clip(gradients, -5, 5, out=gradients) 
	 

	# #Updating the gradient
	# Way = Way - alpha*dWay
	# by = by - alpha*dby
	# Wxa = Wxa - alpha*dWxa
	# Waa = Waa - alpha*dWaa
	# b = b - alpha*db

	# Returning Cross Entropy loss
	return cross_entropy_loss, dWxa, dWaa, dWay,db, dby 


# index for the input data
index_data = [chartoix[i] for i in data]

hl, = plt.plot([], [])
plt.xlim(0, 10000) 
plt.ylim(0, 300)


def draw_lc(hl, error, i):
    hl.set_xdata(np.append(hl.get_xdata(), i))
    hl.set_ydata(np.append(hl.get_ydata(), error))
    plt.draw()

f = open('output.txt','w')

mWxa, mWaa, mWay = np.zeros_like(Wxa), np.zeros_like(Waa), np.zeros_like(Way)
mb, mby = np.zeros_like(b), np.zeros_like(by) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*T # loss at iteration 0

seq_start = 0
a_prev = np.zeros((n_a,1))
epoch = 10000
for i in range(epoch):
	#sample = np.random.randint(0,len(data)-T-1)
	# sample = 500
	if(seq_start >= len(data) - T + 1):
		seq_start = 0
		a_prev = np.zeros((n_a,1))


	index = index_data[seq_start:seq_start+T]

	# index for the output data
	target_data = index_data[seq_start+1:seq_start+T+1]
	target = np.zeros((T,vocab_size))

	# one-hot notation for the output
	target[np.arange(T),target_data] = 1

	#Cross Entropy error
	error, dWxa, dWaa, dWay,db, dby = loss(a_prev,target,index)

	smooth_loss = smooth_loss * 0.999 + error * 0.001
	print("Epoch: " +str(i) + '  ' + str(smooth_loss))

	#parameter update using Adagrad
	for param, dparam, mem in zip([Wxa, Waa, Way, b, by], 
                                [dWxa, dWaa, dWay, db, dby], 
                                [mWxa, mWaa, mWay, mb, mby]):
	    	mem += dparam * dparam
	    	param += -alpha * dparam / np.sqrt(mem + 1e-8) # adagrad update


	if(i%1000==0):
		seq2=[]
		seq = list(np.ravel(np.argmax(p.values(),axis = 0)))
		for k in seq: seq2.append(ixtochar[k])
		f.write(''.join(seq2))
	
	# plt.scatter(i, error, s = 2.5)
	# plt.plot(i,error, linewidth = 3.0)
	draw_lc(hl,smooth_loss,i)
	seq_start = seq_start + T

f.close()
plt.show()







