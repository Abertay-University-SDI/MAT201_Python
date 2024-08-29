#!/usr/bin/env python
# coding: utf-8

# # Iterative Solutions to Equations
# 
# The first numerical technique we will explore are algorithms which find zeros: these are commonly called **root-finding** algorithms. 

# ## Motivation
# 
# The roots of a function are locations/values which make that function equal zero.
# As the name suggests, the roots of a function are often one of the most important properties. Finding the roots of functions is important in many engineering applications such as signal processing and optimization. 
# 
# You have already seen root-finding in action in MAT201: calculating the parameter value when equating a ray with an object requires root-finding. To do this we applied the very common root-finding strategy that you are already familiar with:
# 
# For a quadratic function with the form $f(x) = ax^2 + bx + c$, the ''quadratic formula,'' 
# 
# $$
# x_r=\frac{-b\pm\sqrt{b^2-4ac}}{2a},
# $$
# 
# evaluates the root(s) of the function, $x_r$, exactly (noting that there may be zero, one or two real solutions of this equation). The quadratic formula is a rare example of root-finding with a simple solution, and can only be applied to equations which are *quadratic* (i.e. whose highest power in $x$ is two, $x^2$); most other types of function require novel ways to identify roots. As computational power has significantly grown in recent years, so too have iterative ways to identify roots which take advantage of computational resources and procedures.
# 
# We will here discuss several iterative techinques to identify roots of equations. We will often use quadratic functions to test these tools: we already have an exact analytical way to solve these problems, so this makes a good way to test our numerical approaches and examine how accurate and efficient they are.

# ### Setting up Libraries
# 
# As always, we'll make use of some clever Python tools for plotting and maths, so our first step is to load in the libraries that store these tools:

# In[1]:


## Libraries
import numpy as np
import math 
import sys
get_ipython().system('{sys.executable} -m pip install termcolor')
from termcolor import colored
try:
    from termcolor import colored
except ModuleNotFoundError:
    # Error handling
    import sys
    get_ipython().system('{sys.executable} -m pip install --upgrade termcolor')
    from termcolor import colored
    pass


# (Note that the termcolor package, used to change the font color of some output, doesn't appear to work in Colab. This doesn't really matter, as I was just using it to highlight some specific values).

# 
# ## Simple Iteration 
# In the lectures, we have encountered our first approach to tackling problems in the form $f(x)=0$: **simple iteration**.
# 
# Briefly, simple iteration requires us to rewrite the function in a form with a single dependent variable $x$ on the left hand side, i.e.
# 
# $$
# x=F(x).
# $$
# 
# There may also be several different forms for $F(x)$ per equation.

# In the lectures, we examined this technique using the quadratic equation 
# 
# $$
# x^2-3x+1=0,
# $$
# 
# which has two possible forms of $F(x)$:
# 
# $$
# F_1(x)=\frac{1}{3}\left(x^2+1\right)~~~~{\rm{and}}~~~~ F_2(x)=3-\frac{1}{x}.
# $$
# 
# Lets start by defining some functions to return values according to these equations for a given $x$-value:

# In[2]:


def F1(x):
    result=(x * x + 1.0) / 3.0
    return result
def F2(x):
    result=3.0 - 1.0 / x
    return result     


# We want to:
# - start at a particular value of $x$, 
# - calculate $F(x)$ which then becomes the next value of $x$,
# - repeat the process many times.
# 
# Lets try applying the first few steps, with an initial guess of $x=1$. Using the first function $F_1$, we see:

# In[3]:


x0 = 1.0
x1 = F1(x0)
x2 = F1(x1)
x3 = F1(x2)
x4 = F1(x3)
print(x0, x1, x2, x3, x4)


# As stated in the lectures, this appears to be converging, but its not clear how far we are from convergence. 
# 
# Lets try our second initial guess, $x=3$:

# In[4]:


x0 = 3.0
x1 = F1(x0)
x2 = F1(x1)
x3 = F1(x2)
x4 = F1(x3)
print(x0, x1, x2, x3, x4)


# As per the lectures, this looks to be diverging. Moving on to the second function $F_2$:

# In[5]:


x0 = 1.0
x1 = F2(x0)
x2 = F2(x1)
x3 = F2(x2)
x4 = F2(x3)
print(x0, x1, x2, x3, x4)


# In[6]:


x0 = 3.0
x1 = F2(x0)
x2 = F2(x1)
x3 = F2(x2)
x4 = F2(x3)
print(x0, x1, x2, x3, x4)


# Once again we seem to see $F_2$ converging upon a solution when $x_0=1$ but also converging upon the same solution when $x_0=3$.

# 
# 
# ---
# 
# 
# We'll now explore the test for **convergence**. Remember again from the lectures that a condition for convergence using simple iteration is:
# 
# $$
# \left|F'(x)\right|\lt 1,
# $$
# 
# so we need to test this criteria as we iterate to make sure that we're not carrying out the calculation unecessarily.
# 
# In order to test this, lets create two new functions containing the derivative of the functions $F_1'$ and $F_2'$:

# In[7]:


def dF1dx(x):
    result = 2.0 / 3.0 * x
    return result
def dF2dx(x):
    result = 1.0 / x / x
    return result   


# We have evaluated these derivatives using our mathematical knowledge, but could in principle evaluate these numerically too. We can perform a quick check by evaluating $F'(x_0)$ for both functions:

# In[8]:


temp=[abs(dF1dx(1.0)), abs(dF1dx(3.0)), abs(dF2dx(1.0)), abs(dF2dx(3.0))]
for i in temp:
  mycol='red'
  if float(i) < 1:  #if condition met, switch color to green
    mycol='green' 
  print(colored(i,mycol))


# We can see from this that we would only expect convergence using $F_1$ when $x=1$ and $F_2$ when $x=3$.
# 
# ## Looping Simple Iteration in Python
# 
# So far, we've effectively only used Python as a simple calculator. Python (and other languages) are far more powerful and can potentially check the derivative for convergence/divergence each time an iteration occurs. We can also build in a test of the difference between the latest iteration and previous values in order to not perform the calculation too many times. One could, in principle, combine all of these steps together in the following way:

# In[9]:


x = []
x.append(1.0) #initial function guess
tol = 1e-5
i = 0
while ((len(x) < 50)):  #performing a maximum of 50 iterations for safety
    if ((abs(dF1dx(x[i])) > 1)):  #if divergence expected
      print('divergent')
      break
    x.append(F1(x[i]))  #using F1
    i += 1              #increment i
    if (abs(x[i] - x[i-1]) < tol):  #is the new solution different?
      print('converged')
      break

print("no. of steps:", len(x), "     final x:", x[-1])


# In the above code, we create a list of $x$ values, and place our initial guess for $x$ at the start. The code generates new values in the list, provided a tolerance between successive iterations differs by more than some user defined value (in order to not repeat steps without gaining any extra accuracy).
# 
# We can see with this simple loop that it takes 11 steps from $x_0$=1 to determine the root of the F1 equation to 5dp accuracy. If we shift the initial guess for $x$ to be 3, and repeat the calculation, the "if" statement within the loop detects that the solution is divergent before we carry out any more iterations:

# In[10]:


x = []
x.append(3.0)
i = 0
while ((len(x) < 50)):  #performing a maximum of 50 iterations for safety
    if ((abs(dF1dx(x[i])) > 1)):  #if divergence expected
      print('divergent')
      break
    x.append(F1(x[i]))  #using F1
    i += 1
    if (abs(x[i] - x[i-1]) < tol):  #is the new solution different?
      print('converged')
      break

print("no. of steps:", len(x), "     final x:", x[-1])


# Switching from $F_1$ to $F_2$ in the loop reflects the results we saw earlier (and in the lectures):

# In[11]:


x = []
x.append(1.0)
i = 0
while ((len(x) < 50)):  #performing a maximum of 50 iterations for safety
    if ((abs(dF2dx(x[i])) > 1)):  #if divergence expected
      print('divergent')
      break
    x.append(F2(x[i]))  #using F2
    i += 1
    if (abs(x[i] - x[i-1]) < tol):  #is the new solution different?
      print('converged')
      break

print("no. of steps:", len(x), "     final x:", x[-1])


# Try varying the tolerance and seeing how many iterations the code takes to achieve convergence.

# ## Newton-Raphson Iteration
# 
# Newton-Raphson (also sometimes known as Newton's method) is an iterative technique which rapidly approximates the roots of a real-valued function. It is fast and efficient, yet relatively easy to understand and is in common use in many different practical applications.
# 
# The function we want to work with, $f(x)$, must be smooth and continuous; let us define $x_r$ as an unknown root of $f(x)$. We could make an initial guess $x_0$ for the value of $x_r$. Unless we make a very lucky guess, $f(x_0)$ will not be a root. If we don't find the root, we want our next guess, $x_1$, to be an improvement on $x_0$ (i.e., closer to $x_r$ than $x_0$). If we assume that $x_0$ is "close enough" to $x_r$, then we can improve upon it by taking the linear approximation of $f(x)$ around $x_0$, which is a line, and finding the intersection of this line with the x-axis. Written out, the linear approximation of $f(x)$ around $x_0$ is $f(x) \approx f(x_0) + f^{\prime}(x_0)(x-x_0)$. Using this approximation, we find $x_1$ such that $f(x_1) = 0$. Plugging these values into the linear approximation results in the equation
# 
# $$
# 0 = f(x_0) + f^{\prime}(x_0)(x_1-x_0),
# $$
# 
# which when solved for $x_1$ is
# 
# $$
# x_1 = x_0 - \frac{f(x_0)}{f^{\prime}(x_0)}.
# $$
# 
# An illustration of how this linear approximation improves an initial guess is shown in the following figure.
#  
# 
# <img src="https://pythonnumericalmethods.berkeley.edu/_images/19.04.01-Newton-step.png" alt="Newton Step" title="Illustration of Newton step for a smooth function, g(x)." width="400"/>
# 
# Written generally, a **Newton step** computes an improved guess, $x_i$, using a previous guess $x_{i-1}$, and is given by the equation
# 
# $$
# x_i = x_{i-1} - \frac{g(x_{i-1})}{g^{\prime}(x_{i-1})}.
# $$
# 
# The **Newton-Raphson Method** of finding roots iterates Newton steps from $x_0$ until the error is less than the tolerance.
# 

# Lets return to our earlier function 
# 
# $$
# x^2-3x+1=0.
# $$
# 
# Instead of the simple iterative approach, we can define a function for this expression, and one for its derivative.

# In[12]:


def fnr(x):
    result = x * x - 3 * x + 1
    return result
def dfnrdx(x):
    result = 2 * x - 3 
    return result  


# Now we can calculate each iteration of x based on the previous value and the calls to our two functions. Lets do this for four iterations:

# In[13]:


x = []
x.append(1.0)
x.append(x[0] - fnr(x[0]) / dfnrdx(x[0]))
x.append(x[1] - fnr(x[1]) / dfnrdx(x[1]))
x.append(x[2] - fnr(x[2]) / dfnrdx(x[2]))
x.append(x[3] - fnr(x[3]) / dfnrdx(x[3]))
x.append(x[4] - fnr(x[4]) / dfnrdx(x[4]))
print(x)
x = []
x.append(3.0)
x.append(x[0] - fnr(x[0]) / dfnrdx(x[0]))
x.append(x[1] - fnr(x[1]) / dfnrdx(x[1]))
x.append(x[2] - fnr(x[2]) / dfnrdx(x[2]))
x.append(x[3] - fnr(x[3]) / dfnrdx(x[3]))
x.append(x[4] - fnr(x[4]) / dfnrdx(x[4]))
print(x)


# We can see that this method rapidly converges upon the two roots identified earlier, without needing to worry about divergent solutions.
# 
# Programmers hate unecessarily repeating instructions to a computer - much of what we wrote earlier is simply a repetition of the previous line, so we could simply have the computer repeat that instruction in a loop and stop when the difference between one solution and the next is within a set tolerance: 

# In[14]:


x = []
x.append(1.0)
tol = 1e-5
i = 0
while ((len(x) < 50)):  #performing a maximum of 50 iterations for safety
    x.append(x[i] - fnr(x[i]) / dfnrdx(x[i]))  #using F2
    i += 1
    if (abs(x[i] - x[i-1]) < tol):  #is the new solution different?
      print('converged')
      break
print("ROOT1: no. of steps:", len(x), "     final x:", x[-1])
x = []
x.append(3.0)
i = 0
while ((len(x) < 50)):  #performing a maximum of 50 iterations for safety
    x.append(x[i] - fnr(x[i]) / dfnrdx(x[i]))  #using F2
    i += 1
    if (abs(x[i] - x[i-1]) < tol):  #is the new solution different?
      print('converged')
      break
print("ROOT2: no. of steps:", len(x), "     final x:", x[-1])


# Note that the number of steps taken by Newton-Raphson is far less than the Simple Iteration scheme. As noted in lectures, if there is a root, this scheme converges to the desired degree of accuracy relatively quickly, but still depends (to an extent) on the initial guesses for where the roots are.

# ## Python for mathematics/science
# Python is a really clever language, and lots of equally clever people have extended it, adding libraries to carry out an incredible number of difficult and complicated tasks. One of those libraries, *scipy*, incorporates a root solver, *fsolve*. You can find more information about the documentation [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html).
# Let's call the library:

# In[15]:


from scipy.optimize import fsolve


# Now let's apply it to our problem from earlier. The *fsolve* function takes in several arguments, but for our purposes, we can simply tell it the name of the function to evaluate from earlier, and a list containing our two initial guesses and see what it comes up with:

# In[16]:


fsolve(fnr, [1., 3.])


# Once more we've quickly identified the roots of the equation $x^2-3x+1=0$. There are other root finding methods that we don't have time to cover in this course, but all have positive and negative attributes depending on their application.

# ## Other Examples:
# 
# **TRY IT!** $\sqrt{2}$ is the root of the function $f(x) = x^2 - 2$. Using $x_0 = 1.4$ as a starting point, use the information you've seen in this worksheet to estimate $\sqrt{2}$. Can you recycle some of the earlier algorithmns to do the hard work for you?
# 
# Compare this approximation with the value computed by Python's sqrt function.
# 
# $$
# x = 1.4 - \frac{1.4^2 - 2}{2(1.4)} = 1.4142857142857144
# $$

# In[ ]:




