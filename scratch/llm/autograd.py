"""Lesson 1 - autograd: how a neural network learns, in plain Python (no PyTorch, no NumPy).

    python autograd.py

The whole of deep learning rests on one loop:
    1. forward pass   - run the numbers through the network and get a prediction
    2. loss           - one number that says how wrong the prediction is
    3. backward pass  - for every weight, work out "if I nudge this weight up, does the loss go up or down?"
                        That number is the gradient.
    4. update         - move every weight a small step against its gradient (gradient descent)

Step 3 is backpropagation. It is only the chain rule from calculus, applied to a graph of small operations.
PyTorch does exactly this for tensors. Here we do it for single numbers so nothing is hidden.
"""

import math
import random


class Value:
    """One number that remembers how it was made, so the gradient can flow back through it."""

    def __init__(self, data: float, parents: tuple = ()) -> None:
        self.data = data
        self.grad = 0.0  # d(loss)/d(this value). Filled in by backward().
        self._parents = parents  # the Values this one was computed from: the computation graph
        self._backward = lambda: None  # pushes this node's grad to its parents (the local chain rule step)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            # d(a+b)/da = 1 and d(a+b)/db = 1, so the gradient passes through unchanged.
            # "+=" and not "=": a value used in two places collects gradient from both.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            # d(a*b)/da = b and d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float):
        out = Value(self.data**k, (self,))

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad  # power rule

        out._backward = _backward
        return out

    def tanh(self):
        """An activation function: squashes any number into (-1, 1). Without one, stacked layers collapse
        into a single linear layer and the network cannot learn curves."""
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagation. Visit every node after all the nodes that depend on it (reverse topological order)
        and let each one hand its gradient to its parents."""
        order, seen = [], set()

        def visit(v: "Value") -> None:
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)

        visit(self)
        self.grad = 1.0  # d(loss)/d(loss) = 1
        for v in reversed(order):
            v._backward()

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)


class Neuron:
    """output = tanh(w1*x1 + w2*x2 + ... + bias). The weights and the bias are the parameters we learn."""

    def __init__(self, n_in: int, activate: bool = True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]  # random init breaks the symmetry
        self.b = Value(0.0)
        self.activate = activate

    def __call__(self, x):
        s = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return s.tanh() if self.activate else s

    def parameters(self):
        return self.w + [self.b]


class MLP:
    """Multi-layer perceptron: layers of neurons, each layer feeding the next. The same thing sits inside
    every transformer block (see model.py: FeedForward)."""

    def __init__(self, sizes: list[int]) -> None:
        self.layers = [[Neuron(a, activate=i < len(sizes) - 2) for _ in range(b)]
                       for i, (a, b) in enumerate(zip(sizes, sizes[1:]))]

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


def demo() -> None:
    # Check 1: our gradient matches the "nudge it and see" (finite difference) gradient.
    def f(a, b):
        return (a * b + a**2).tanh()

    a, b = Value(0.7), Value(-1.3)
    f(a, b).backward()
    h = 1e-6
    numeric = (f(Value(0.7 + h), Value(-1.3)).data - f(Value(0.7 - h), Value(-1.3)).data) / (2 * h)
    assert abs(a.grad - numeric) < 1e-6, (a.grad, numeric)
    print(f"gradient check   backprop {a.grad:.6f}   numeric {numeric:.6f}")

    # Check 2: a tiny network learns XOR, a problem no single linear layer can solve.
    random.seed(0)
    xs, ys = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
    net = MLP([2, 8, 1])
    lr = 0.05  # learning rate: the size of each step. Too big and the loss explodes (divergence).
    for step in range(1000):
        loss = sum(((net(x) - y) ** 2 for x, y in zip(xs, ys)), Value(0.0)) * 0.25  # mean squared error
        for p in net.parameters():
            p.grad = 0.0  # gradients add up, so clear them before every backward pass
        loss.backward()
        for p in net.parameters():
            p.data -= lr * p.grad  # stochastic gradient descent (SGD)
        if step % 200 == 0:
            print(f"step {step:4d}  loss {loss.data:.4f}")
    print("XOR predictions", [round(net(x).data, 2) for x in xs], "target", ys)
    assert loss.data < 0.05, "the network did not learn XOR"


if __name__ == "__main__":
    demo()
