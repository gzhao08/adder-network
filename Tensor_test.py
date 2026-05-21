from Tensor import Tensor
import numpy as np

x = np.array([[1,2],[3,4],[5,6]])
x = Tensor(x)

y = np.array([[0.5],[0.5]])
y = Tensor(y)

a = np.array([[1,2,3]])
a = Tensor(a)

z = x@y

b = a@z
print(f"b=\n{b}")
b.backward()

print(f"x=\n{z}")
# print(y)
# print(z)

a = np.array([[5]])
b = np.array([[1,2],[3,4],[5,6]])

a = Tensor(a)
b = Tensor(b)
c = 5*b
c.backward()
print(b)