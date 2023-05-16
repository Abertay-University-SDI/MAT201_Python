#!/usr/bin/env python
# coding: utf-8

# # Complex Numbers
# JT 2023. Ongoing MAT201 expansion notebooks.

# In[1]:


## Libraries
import numpy as np
import math 
import matplotlib.pyplot as plt
import sympy as sym


# ##Introduction
# As seen in the lectures, complex numbers comprise of a real part and an imaginary part, with the letter "i" indicating which is which. In general a complex number is denoted as
# 
# $$
# z = A+iB,
# $$
# 
# where $A$ is the real part, $B$ is the imaginary part and $i=\sqrt{-1}$. 
# 
# To represent complex numbers in Python, there are two approaches. Python will interpret a number as complex if we append the second value in a sum with the letter j (j is a more common letter in engineering to represent the square-root of -1; i is used, for example, to represent electric current in physics equations): 

# In[2]:


z1 = 1 + 2j
z2 = -2 - 2j


# ## Basic arithmetic operations
# 
# When we add or subtract complex numbers, we must add or subtract the real parts and imaginary components separately:
# 
# $$
# (A+iB)+(C+iD)=(A+C)+i(B+D).
# $$
# 
# Python understands this too:

# In[3]:


print("z1+z2=",z1+z2)
print("z1-z2=",z1-z2)


# When we multiply complex numbers, we create a product of four terms:
# 
# $$
# \begin{align}
# (A+iB)⋅(C+iD)&=A \cdot C+iB\cdot C+A\cdot iD+iB\cdot iD. \\
# &=AC+i(BC+AD)+i^2BD=(AC-BD)+i(BC+AD)
# \end{align}
# $$
# 
# When both terms containing $i$ multiply, we need to remember $i^2={\sqrt{-1}}^2=-1$. Python also knows that this is the case.
# 
# For example:
# $$
# \begin{align}
# (1+2i)⋅(-2-2i)&=1 \cdot -2+2i\cdot -2+1\cdot -2i+2i\cdot -2i. \\
# &=-2+i(-4-2)-4i^2=(-2+4)-6i=2-6i
# \end{align}
# $$
# 
# Let's check using Python:

# In[4]:


print("z1*z2=",z1*z2)


# If we want to do some more advanced things with complex numbers in Python, we should use the "complex" datatype. This tells Python that we are serious about complex numbers and allows us access to more useful mathematical operations related to complex numbers. One example is the ability to immediately tell us what the conjugate of any given complex number is. 
# 
# From the lectures, we learned that the conjugate of a complex number, $\bar{z}$, has an imaginary component of opposite sign:
# 
# $$
# z=A+iB,\\
# \bar{z}=A-iB.
# $$
# 

# In[5]:


z3 = complex(3.,1.)
print("z3= ",z3)
print("conj(z3)=",z3.conjugate())


# Remember too that any complex number multiplied by it's conjugate must entirely be real (the two imaginary products cancel, while the $i^2$ term becomes real).

# In[6]:


print("z3*conj(z3) = ",z3*z3.conjugate())


# ## Complex number division
# 
# The complex conjugate is really useful for dividing complex numbers. We can eliminate the presence of a complex term in the denominator by multiplying it by it's conjugate; to balance this, we also multiply the numerator by the complex conjugate of the denominator.
# 
# We can confirm this approach by having Python carry out the complex division in one go, then checking that it is equivalent to the approach we saw in lectures:

# In[7]:


print("z2/z3=",z2/z3)
numerator = z2*z3.conjugate()
denominator = z3*z3.conjugate()
print("ans=", numerator, "/", denominator, "=", numerator/denominator)


# ## Visualising complex numbers
# One way to visualise complex numbers is to use the Argand diagram. In this diagram we plot the real component of the complex number as the $x$-coordinate, and the imaginary component as the $y$-coordinate.
# 
# To show this, we'll define a function called "argand". The argument of this function is a complex number. Our function will return the location of this complex number on the Argand diagram, highlighting the $x$- and $y$-components:

# In[8]:


def argand(a):
    """Function to create an argand diagram for a complex number a.
  """
    for x in range(len(a)):
        plt.plot([0,a[x].real],[0,a[x].imag],'ro-',label='python')
        plt.text(a[x].real+0.15, a[x].imag+0.15, a[x], c='blue')
        plt.plot([a[x].real,a[x].real],[0,a[x].imag],'k--',label='python')
        plt.text(a[x].real+0.15, 0.5*a[x].imag+0.2, a[x].imag, c='k')
        plt.plot([0,a[x].real],[0,0],'k--',label='python')
        plt.text(0.5*a[x].real, -0.25, a[x].real, c='k')
    limit=np.max(np.ceil(np.absolute(a))) # set limits for axis
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.show()


# Lets see what happens if we test this with one of our earlier complex numbers:

# In[9]:


argand([z3])


# You can see from the plot that the complex number $3+1j$ is represented by travelling $3$ along in $x$ and $1$ up in $y$. This form of complex number is sometimes known as "Cartesian form". One can alternatively express complex numbers in polar form, or "modulus and argument" form. For a complex number $z=A+iB$:
# 
# 
# *   The modulus of a complex number is the size of the line linking the Cartesian position and the origin, and is calculated as $|z|=\sqrt{A^2+B^2}$.
# *   The argument of a complex number is the angle subtended between the line linking the complex number and the origin, and the positive real axis. 
# 
# The argument, $\theta$, is calculated according to 
# 
# $$
# \theta=\tan^{-1}{\left(\frac{|B|}{|A|}\right)},
# $$ 
# 
# when located in the positive $x$ and positive $y$ part of the diagram. If either $B$ or $A$ is positive one must add/subtract in order to form the argument of the complex number relative to the positive real axis.
# 
# Most of the angle calculation methods baked into Python already take account of this, using the *atan2* function, or the *angle* command from numpy:
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[10]:


argz3 = np.arctan2(z3.imag, z3.real)
print(math.degrees(argz3))
print(np.angle(z3, deg=True))
print(np.angle(complex(-1,2), deg=True))


# *atan2* is a fairly common function in several computer languages: you can find more information about it through e.g. Google and other search engines. The numpy command *angle* can handle complex datatypes as input, which can save use some effort compared to *atan2*:

# In[11]:


print(np.angle(0+1j,deg=True))


# ## Cmath
# Python has an explicit package solely for manipulating complex numbers, called *cmath*. It goes far beyond our needs (complex numbers are an entire interesting branch of mathematics with wide-ranging implications and applications). Some of the more basic functions can be useful however: *cmath* can quickly and directly convert to/from polar form, for example using the *polar* command:

# In[12]:


#import cmath for complex number operations
import cmath

#find the polar coordinates of complex number
print(cmath.polar(2 + 3j))
print(cmath.polar(1 + 5j))


# ## Over to you!
# These examples are one way that we can use Python to illustrate some of our methods taught in lectures.
# 
# Can you confirm some of the tutorial question solutions using Python? Can you plot these on the Argand diagram, or evaluate them in polar form?

# In[ ]:




