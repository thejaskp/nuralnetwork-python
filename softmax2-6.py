import math
import numpy as np



layer_outputs = [[4.8, 1.21, 2.38],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

#keep in mind that to overcome the overflow in softmax acticvation function we will sumstracting the bigger value from our input to all the values so our range will be from 0 to 1.


norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)




# print(norm_values)
# print(sum(norm_values))