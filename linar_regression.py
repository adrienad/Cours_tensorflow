import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(10,6)

X=np.arange(0.0, 5.0, 0.1)

#a=1
#b=0
#
#Y=a*X+b
#
#plt.plot(X,Y)
#plt.ylabel('Dependent Variable')
#plt.xlabel('Independent Variable')
#plt.show()

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*3+2

#np.random.normal=draw random samples from a normal (Gaussian) distribution
#lambda=creation of anonymous function 
#ainsi la ligne en dessous definie une fonction gaussienne qui ajoute un nombre
#aleatoire suivant une fonction gaussienne et l'applique à y_data
y_data=np.vectorize(lambda y: y+ np.random.normal(loc=0.0, scale=0.1))(y_data)

#zip is a function that agregate elements from each of the iterables