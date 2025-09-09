import tensorflow as tf


# Creating Tensors
string1 = tf.Variable("this is a string", tf.string) 
number1 = tf.Variable(324, tf.int16)
floating1 = tf.Variable(3.567, tf.float64)


# Rank/Degree of Tensors
rank1_tensor = tf.Variable(["Test","sdsd"], tf.string) 
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)


# Changing Shape
tensor1 = tf.ones([1,2,3])  
tensor2 = tf.reshape(tensor1, [2,3,1])  
tensor3 = tf.reshape(tensor2, [3, -1])
