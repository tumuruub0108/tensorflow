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


# Scalars (0D)
a = tf.constant(5)
b = tf.constant(2)

print("Scalar addition:", a + b)


# Vectors (1D)
v1 = tf.constant([1, 2, 3])
v2 = tf.constant([4, 5, 6])

print("Vector addition:", v1 + v2)


# Matrices (2D)
m1 = tf.constant([[1, 2], [3, 4]])
m2 = tf.constant([[5, 6], [7, 8]])

print("Matrix multiplication:\n", tf.matmul(m1, m2))


# 3D Tensor (batch of matrices)
t1 = tf.constant([
    [[1, 2], [3, 4]],    # first matrix
    [[5, 6], [7, 8]]     # second matrix
])
t2 = tf.constant([
    [[1, 0], [0, 1]],    # identity matrix
    [[2, 0], [0, 2]]     # scaled identity
])

print("Batch matrix multiplication:\n", tf.matmul(t1, t2))
