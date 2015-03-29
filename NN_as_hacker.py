## Python verstion (code) of  Hacker's guide to Neural Networks

##http://karpathy.github.io/neuralnets/

##Chapter 1: Real-valued Circuits

def forwardMultiplyGate(x,y): return x * y

print forwardMultiplyGate(-2, 3);  # returns -6. Exciting.


 #Strategy #1: Random Local Search
from random import uniform
from decimal import Decimal


 ## circuit with single gate for now
def forwardMultiplyGate(x,y): return x * y

x , y = -2, 3 ##some input values

## try changing x,y randomly small amounts and keep track of what works best

tweak_amount = 0.01

best_out = Decimal('-Infinity')

best_x , best_y = x, y

for k in range(100) :

  x_try = x + tweak_amount * (uniform(0,1) * 2 - 1)  #tweak x a bit
  y_try = y + tweak_amount * (uniform(0,1) * 2 - 1) #tweak y a bit

  out = forwardMultiplyGate(x_try, y_try)

  if out > best_out:
    ## best improvement yet! Keep track of the x and y
    best_out = out

    best_x, best_y = x_try, y_try

print "Best value of x is {} and Best value of y is {}.".format(best_x,best_y)
print "So best value for the function is {}.".format(forwardMultiplyGate(best_x,best_y))


## Stategy #2: Numerical Gradient

x , y = -2, 3 ##some input values

out = forwardMultiplyGate(x, y) ## -6
print out

h = 0.0001;

##compute derivative with respect to x

xph = x + h; ##-1.9999

out2 = forwardMultiplyGate(xph, y)  ## -5.9997
x_derivative = (out2 - out) / h     ## 3.0
print x_derivative
## compute derivative with respect to y

yph = y + h  ##3.0001

out3 = forwardMultiplyGate(x, yph) # -6.0002
y_derivative = (out3 - out) / h 
print y_derivative


## gradient a tiny amount

step_size = 0.01;
out = forwardMultiplyGate(x, y) 	## before: -6
print out
x = x + step_size * x_derivative   	## x becomes -1.97
print x
y = y + step_size * y_derivative 	## y becomes 2.98
print y
out_new = forwardMultiplyGate(x, y) ## -5.87! exciting.
print out_new


##Strategy #3: Analytic Gradient

x , y = -2, 3 ##some input values

out = forwardMultiplyGate(x, y)  ##before: -6
print out

x_gradient = y   ##by our complex mathematical derivation above

y_gradient = x

step_size = 0.01

x += step_size * x_gradient; 	##-2.03
print x
y += step_size * y_gradient; 	##2.98
print y

out_new = forwardMultiplyGate(x, y)  ## -5.87. Higher output! Nice.
print out_new

##Recursive Case: Circuits with Multiple Gates

def forwardMultiplyGate (a, b):  return a * b

def forwardAddGate(a, b):  return a + b

def forwardCircuit(x,y,z): 
  q = forwardAddGate(x, y)
  f = forwardMultiplyGate(q, z)
  return f

x , y , z = -2, 5,-4  ## Multiple variable assignment

f = forwardCircuit(x, y, z)  ## output is -12
print f


##  Backpropagation

##initial conditions
x , y , z = -2, 5,-4  ## Multiple variable assignment
q = forwardAddGate(x, y)		## q is 3
f = forwardMultiplyGate(q, z)	## output is -12

## gradient of the MULTIPLY gate with respect to its inputs
## wrt is short for "with respect to"
derivative_f_wrt_z = q 		## 3
derivative_f_wrt_q = z  	##-4

## derivative of the ADD gate with respect to its inputs
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

## chain rule
derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q	## -4
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q 	## -4


## final gradient, from above: [-4, -4, 3]

gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

##  let the inputs respond to the force/tug:
step_size = 0.01
x = x + step_size * derivative_f_wrt_x	## -2.04
y = y + step_size * derivative_f_wrt_y	##  4.96
z = z + step_size * derivative_f_wrt_z	## -3.97

## Our circuit now better give higher output:
q = forwardAddGate(x, y) 	## q becomes 2.92
f = forwardMultiplyGate(q, z)	## output is -11.59, up from -12! Nice!

## sabutt check for chain rule with numerical gradient

## initial conditions

x , y , z = -2, 5,-4  ## Multiple variable assignment

##  numerical gradient check
h = 0.0001

x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h  	## -4
y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h 	## -4
z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h 	##  3

print x_derivative, y_derivative, z_derivative

"""
## Example: Single Neuron

## every Unit corresponds to a wire in the diagrams

class Unit:

	def __init__(this,value, grad):

		##value computed in the forward pass
		this.value = value
		##the derivative of circuit output w.r.t this unit, computed in backward pass
		this.grad  = grad


class multiplyGate:

	def forward(this, u0 ,u1):
		'''
		store pointers to input Units u0 and u1 and output unit utop
		'''
  		this.u0 = u0
    	this.u1 = u1
    	this.utop = Unit(u0 * u1, 0.0)
    	return this.utop

	def backward(this):

    	##take the gradient in output unit and chain it with the
    	##local gradients, which we derived for multiply gate before
    	##then write those gradients to those Units.    
    	this.u0.grad += this.u1 * this.utop.grad
    	this.u1.grad += this.u0.value * this.utop.grad
"""

