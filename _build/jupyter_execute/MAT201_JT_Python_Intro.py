#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction to Python
# 
# Brief introduction to basic Python syntax, libraries, plotting in Jupyter notebooks.
# 

# ## General introduction
# 
# This notebook lists examples of simple Python commands and programs that can be used to help with the understanding of the MAT201 course, particularly in the area of dynamics. Students can use Python to check their answers and visually see the mathematics in action. More broadly, a wide knowledge and ability to use other programming languages may help with problem solving and employability. 
# 
# This is a Jupyter notebook launched from within the Anaconda data science platform. Anaconda can be downloaded here:
# 
# [https://www.anaconda.com](https://www.anaconda.com)
# 
# Python is an open-source programming language used extensively around the world. It is also vital for carrying out Scientific Computation.
# 
# To execute commands in Jupyter, either Run the cell or hit SHIFT + ENTER.
# 
# Alternatively, you can perform cloud computing using Google Colab:
# 
# [https://colab.research.google.com/](https://colab.research.google.com/)
# 
# You will need a Google account to perform cloud computing, but the commands are (largely) the same. One can also click the "play" button next to an executable segment of code to run that code.

# Notebooks in Jupyter look like this: ![title](JupyterNotebook_my_first_plot.png)

# In[1]:


print("Hello, World")


# The output should always appear below each code cell (if there is any).
# 
# Note if you hit ENTER without the shift, the notebook will either open the cell for editing, or perform a carriage return within that cell instead. You can edit code or text, then run it with SHIFT + ENTER.
# 
# Try modifying this statement yourself: **Python is ....**
# 
# I encourage you to make copies of notebooks (to augment your lecture notes) and save them to your own computer or OneDrive areas. 

# Python has extended functionality through libraries containing lots of extra commands, functions, and useful stuff. We import these libraries and commands using the *import* command:

# In[2]:


import time
time.sleep(3)


# In this case, we have imported a library called "time" in order to run a command called "sleep". This command runs for an amount of time given by its argument (in our case, 3 seconds) and produces no output. Note that when the code is running the symbol within the square brackets changes (indicating it is running).

# There are **many** detailed introductions to Jupyter notebooks, and Google colab, including textbooks, online workshops, videos, notebooks, theatrical productions (well maybe)... most are available on the web, like
# [this one](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). Youtube videos [like this one](https://youtu.be/RLYoEyIHL6A) also allow you to see this process as you would on your machine.
# 
# 
# 
# ---
# 
# 

# ## Applications to MAT201 content
# 
# In MAT201, our focus is (as you might expect) on mathematics. The rest of this introduction will illustrate a Python library called [Sympy](https://www.sympy.org/en/index.html). Like other Python packages, it is well documented, and widely used
# 
# Let's go ahead and load in the libraries we need: in this case, we'll focus solely on *sympy*:
# 

# In[3]:


import sympy as sym


# Let's very briefly look at some capabilities of *sympy*. *Sympy* has three data types, Real Integer and Rational. Rational numbers are probably better known as fractions. If we wanted to work with the fraction of one half, for example, we could write this as a real number $0.5$, or as a fraction:

# In[4]:


onehalf = sym.Rational(1,2)
print(onehalf)


# We can perform basic maths with symbols in the usual way (e.g. addition, subtraction, multiplication, division):

# In[5]:


print("4*onehalf=", 4 * onehalf)


# We can use symbolic constants like $pi$, $e$ (as in the exponential) or indeed infinity:

# In[6]:


mypi = sym.pi
print(mypi.evalf())


# This is just for starters: we can build systems of equations if we tell Python that some symbols represent unknown numbers:

# In[7]:


x, y = sym.Symbol('x'), sym.Symbol('y')
print(x + 2 * x + onehalf)


# See how Python has simplified the above expression, and added together the x terms? We can get it to do more complicated algebra:

# In[8]:


sym.expand( (x + onehalf * y)**2 )


# Sympy has expanded this binomial, just like we would using pen and paper. It knows all the rules we do! It also knows lots about trigonometric functions. The *simplify* command can simplify expressions, using all of the maths knowledge in the library. We know that $\cos^2{(x)}+\sin^2{(x)}=1$; so does sympy:

# In[9]:


print( sym.cos(x) * sym.cos(x) + sym.sin(x) * sym.sin(x))
print( sym.simplify( sym.cos(x) * sym.cos(x) + sym.sin(x) * sym.sin(x)) )


# It can even perform calculus operations like differentiation and integration:

# In[10]:


sym.diff(onehalf * x ** 2, x)


# where the command has differentiated the first expression with respect to x. We can perform partial derivatives just as easily: lets say we need \begin{equation} 
# \frac{\partial}{\partial x} \left(x^3+y+axy\right), 
# \end{equation}
# where $x$ and $y$ are independent variables:

# In[11]:


sym.diff(x ** 3 + y + onehalf * x * y , x)


# easy peasy! What about integration? We know that (ignoring the constants of integration for now) 
# \begin{equation}\int \frac{1}{{x}} ~{\rm{d}}x=\ln{|x|}+c,\end{equation} but so does sympy:

# In[12]:


sym.integrate(1 / x, x)


# (also noting that sympy refers to natural log ($ln$) as "log"). Sympy can also solve equations or sets of equations in one line:

# In[13]:


solution = sym.solve((x + 5 * y - 2, -3 * x + 6 * y - 15), (x, y))
solution[x], solution[y]


# It can also solve matrix equations and tons of other clever things. However, this is plenty of information to begin to explore simple differential calculus that we'll encounter in our MAT201 dynamics lectures.

# In[ ]:




