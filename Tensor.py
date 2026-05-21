import numpy as np

class Tensor:

    def __init__(self, data, _children=(), _op='', label=''):
        """
        @brief Construct a Tensor node.

        @param data      The scalar numeric Tensor.
        @param _children Tuple of Tensor nodes used to compute this Tensor.
        @param _op       String describing the operation that produced this node.
        @param label     Optional label for visualization/debugging.
        """

        ## Stored scalar Tensor
        self.data = data

        ## Gradient of output with respect to this Tensor
        self.grad = np.zeros(data.shape)

        ## Function used during backpropagation
        self._backward = lambda: None

        ## Set of parent nodes in the computational graph
        self._prev = set(_children)

        ## Operation that produced this node (for graph visualization)
        self._op = _op

        ## Optional label
        self.label = label

    def __repr__(self):
        """
        @brief String representation of the Tensor object.

        @return String showing the stored scalar Tensor.
        """

        return (
            f"Tensor(\n"
            f"data=\n"
            f"{np.round(self.data, 2)},\n"
            f"grad=\n"
            f"{np.round(self.grad, 2)}\n"
            f")"
        )
    


    def __add__(self, other):
        """
        @brief Addition operator overload.

        Builds a new computational graph node representing:
            out = self + other

        @param other Tensor or scalar to add.
        @return Tensor node representing the result.
        """

        other = other if isinstance(other, Tensor) else Tensor(other*np.ones(self.data.shape))
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1
            self.grad += 1.0 * out.grad
            # d(out)/d(other) = 1
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out


    def __mul__(self, other):

        other = Tensor(np.array([[other]]))
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule derivatives
            self.grad += other.data * out.grad
            other.grad += np.sum(self.data * out.grad)

        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad 

        out._backward = _backward
        return out


    def __pow__(self, other):
        """
        @brief Power operator overload.

        Computes:
            out = self ** other

        @param other Exponent (must be int or float).
        @return Tensor node representing the result.
        """

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            # d/dx (x^n) = n*x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out


    def __rmul__(self, other):
        """
        @brief Right multiplication overload.

        Enables expressions like:
            scalar * Tensor
        """
        return self * other


    def __truediv__(self, other):
        """
        @brief Division operator overload.

        Computes:
            self / other
        """
        return self * other**-1


    def __neg__(self):
        """
        @brief Unary negation operator.

        Computes:
            -self
        """
        return self * -1


    def __sub__(self, other):
        """
        @brief Subtraction operator overload.

        Computes:
            self - other
        """
        return self + (-other)


    def __radd__(self, other):
        """
        @brief Right addition overload.

        Enables expressions like:
            scalar + Tensor
        """
        return self + other
    
    def __abs__(self):
        """
        @brief Absolute Tensor

        Enables expressions like:
            |Tensor|
        """
        out = Tensor(abs(self.data),(self, ), 'abs')

        def _backward():
            # d/dx |x| = x for use in adderNet (an approximation)
            self.grad += np.clip(self.data,-1,1) * out.grad
        
        out._backward = _backward
        return out


    def tanh(self):
        """
        @brief Hyperbolic tangent activation function.

        Computes:
            tanh(x)

        @return Tensor node representing tanh(self).
        """

        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (self, ), 'tanh')

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


    def exp(self):
        """
        @brief Exponential function.

        Computes:
            e^x

        @return Tensor node representing exp(self).
        """

        x = self.data
        out = Tensor(math.exp(x), (self, ), 'exp')

        def _backward():
            # d/dx e^x = e^x
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        """
        @brief Logarithm function.

        Computes:
            log(x)

        @return Tensor node representing log(self).
        """

        x = self.data
        out = Tensor(math.log(x), (self, ), 'exp')

        def _backward():
            # d/dx log(x) = 1/x
            self.grad += out.grad / x

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(max(0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out



    def backward(self):
        """
        @brief Perform reverse-mode automatic differentiation.

        Computes gradients of all nodes in the computational graph
        with respect to this Tensor (assumed to be the output node).

        Steps:
        1. Build a topologically sorted list of nodes.
        2. Initialize output gradient to 1.
        3. Traverse graph in reverse order calling each node's
        stored backward function.
        """

        topo = []
        visited = set()

        def build_topo(v):
            """
            @brief Recursively build a topological ordering of the graph.
            """
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient
        self.grad = np.ones(self.data.shape)

        # Backpropagate
        for node in reversed(topo):
            node._backward()
