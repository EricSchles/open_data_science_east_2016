##Understanding Neural Networks

Neural networks - usually thought of as black box solutions to solving problems, need not be viewed in that light any further.  While building up the intuition for what happens during execution of a neural network will likely continue to remain challenging for at least a little while, an important first step is understanding how to write your own neural network.  By analyzing and understanding the algorithms that go into this technique (since there is no hard intuition yet) we can at least start to understand what's going on and why.

At the heart of neural networks is a technique called stochastic gradient descent.  Neural networks primarily operate over matrices and so the implementation of SGD can be challenging.  In order to draw the necessary intuition about our model let's look at another technique that makes use of SGD - newton's method:

Newton's method is so simple to implement we're going to look at it first and then understand why it works:

```
from hackthederivative import complex_step_finite_diff as deriv

def dx(f,x):
    return abs(0-f(x))

def newtons_method(f,x0,e):
    delta = dx(f,x0)
    while delta > e:
        x0 = (x0 - f(x0))/deriv(f,x0)
        delta = dx(f,x0)
    print("Root is at: ", x0)
    print("f(x) at root is: ", f(x0))


roots = [newtons_method(lambda x: 6*x**5 - 5*x**4 - 4*x**3 + 3*x**2,i,1e-5) for i in range(1000)]
```

So what's the goal?  To find all the roots of the function.  To do this we look at the function 