import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(10,6)



#X=np.arange(0.0, 5.0, 0.1)
#a=1
#b=0
#
#Y=a*X+b
#
#plt.plot(X,Y)
#plt.ylabel('Dependent Variable')
#plt.xlabel('Independent Variable')
#plt.show()


#np.random.rand est 
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*3+2

#np.random.normal=draw random samples from a normal (Gaussian) distribution
#lambda=creation of anonymous function 
#ainsi la ligne en dessous definie une fonction gaussienne qui ajoute un nombre
#aleatoire suivant une fonction gaussienne et l'applique à y_data
y_data=np.vectorize(lambda y: y+ np.random.normal(loc=0.0, scale=0.1))(y_data)

#zip is a function that agregate elements from each of the iterables

#zip(x_data,y_data) [0:5]

a=tf.Variable(1.0)
b=tf.Variable(0.2)

y=a*x_data+b

#loss=value/error that will be minimized during training
#tf.reduce_mean find the mean of a multidimentional tensor (the dimensions can change)
loss=tf.reduce_mean(tf.square(y-y_data))

#learning rate=0,5
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

#start running optimization process
train_data=[]
for step in range(100):
    evals=sess.run([train,a,b])[1:]
    if step % 5 ==0:
        print(step,evals)
        train_data.append(evals)

convert=plt.colors
cr, cg, cb= (1.0, 1.0, 0.0)
for f in train_data:
    cb+=1.0/len(train_data)
    cg-=1.0/len(train_data)
    if cb>1.0: cb = 1.0
    if cg< 0.0: cg = 0.0
    [a,b]=f
    f_y=np.vectorize(lambda x: a*x +b)(x_data)
    line=plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')


green_line = mpatches.patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()

    

