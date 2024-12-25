from __future__ import print_function
import tensorflow as tf

# 定义张量常数
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)

# 各种张量运算
# 注意：张量支持python操作符，如+、-、*、/等
add = tf.add(a, b) # 加法
sub = tf.subtract(a, b) # 减法
mul = tf.multiply(a, b) # 乘法
div = tf.divide(a, b) # 除法

# 获取张量的值
print("-------------------------------")
print("add =" , add.numpy())
print("sub =" , sub.numpy())
print("mul =" , mul.numpy())
print("div =" , div.numpy())
print("-------------------------------")

# 更多的操作
mean = tf.reduce_mean([a, b, c]) # 平均值
sum = tf.reduce_sum([a, b, c]) # 总和

# 获取张量的值
print("mean =" , mean.numpy())
print("sum =" , sum.numpy())
print("-------------------------------")

# 矩阵乘法
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])

product = tf.matmul(matrix1, matrix2) # 矩阵乘法

# 获取张量的值
print("product =" , product.numpy())
print("-------------------------------")