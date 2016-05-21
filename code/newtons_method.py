from hackthederivative import complex_step_finite_diff as deriv

def dx(f,x):
    return abs(0-f(x))

def newtons_method(f,x0,e):
    delta = dx(f,x0)
    while delta > e:
        x0 = x0 - f(x0)/deriv(f,x0)
        delta = dx(f,x0)
    return x0

f = lambda x: 6*x**5 - 5*x**4 - 4*x**3 + 3*x**2
roots = [newtons_method(lambda x: 6*x**5 - 5*x**4 - 4*x**3 + 3*x**2,i,1e-5) for i in range(10)]
print("Roots:",roots)
print("Function at Roots:",[f(root) for root in roots])
