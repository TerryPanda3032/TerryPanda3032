import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import*
from keras import layers
import numpy as np
from keras import regularizers
####################################
print(" import==100%")
Axis_x=np.linspace(0, 100,100)
Axis_y=0
Axis_y=3*Axis_x + 6 + np.random.randn(100) * 7
print(Axis_x,Axis_y)
plt.plot(Axis_x,Axis_y)
####################################
model=keras.Sequential()
model.add(layers.Dense(1,input_dim=1))
model.summary()
######################################
model.compile(optimizer="adam", loss="mse")
#######################################
model.fit(Axis_x,Axis_y,epochs=8000)
#####################################
plt.plot(Axis_x,Axis_y)
plt.scatter(Axis_x,Axis_y)
x=model.predict(Axis_x)
y=model.predict(Axis_y)
plt.plot(x, y)
plt.scatter(x,y)
plt.show()
a=model.predict([100])
plt.scatter(a, a)
plt.show()






