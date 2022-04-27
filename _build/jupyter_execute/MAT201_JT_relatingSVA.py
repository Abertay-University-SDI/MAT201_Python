#!/usr/bin/env python
# coding: utf-8

# # Displacement, Velocity and Acceleration by Integration & Differentiation
# 
# Interactive code to illustrate concepts from MAT201 at Abertay University (JT 2022).
# 
# Using Python's symbolic computation library, we will explore how displacement, velocity and acceleration are related, first in the case of uniform acceleration.
# 
# If you have not already done so, please check out the MAT201 Python introduction notebook to better understand the content shown here.

# In Lecture 1 of our Dynamics component of MAT201, we saw how to relate velocity, displacement and acceleration mathematically. Lecture two also focusses on expanding on these ideas, using Newton's Second Law to relate forces acting on objects with non-constant acceleration. The ultimate goal of both lectures is the same: given some information about an object, can you determine other related properties/behaviour using mathematics? 
# 
# We'll repeat some of the examples we carried out in Lectures 1 and 2, with the help of Python. To do this, we'll require a library for symbolic computation called *sympy*. This library lets Python play with symbols, rather than numbers, and even solve algebraic equations. The first section in this worksheet will briefly demonstrate some key features of sympy (though more extensive documentation can easily be found online). Section 2 will explore applying these tools to uniform acceleration (Lecture 1) while Section 3 will focus on the behaviour of an object subject to a resistive force (non-constant acceleration, Lecture 2).

# Let's go ahead and load in the libraries we need: in this case, we'll focus solely on sympy:
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sympy as sym


# ## Relating acceleration, velocity and displacement by differentiation and integration:
# 
# In the lectures we saw that 
# \begin{equation}
# \int^s_{s_0} {\rm d}x = \int^t_0 v~{\rm{d}}t.
# \end{equation}
# This means that if we know the velocity, expressed as a function of time, we can **integrate** this to determine the displacement of the object (relative to its initial displacement). The constants of integration determine the range over which we're interested, usually between a time $t=$ (when the displacement is $s_0$) to some other time $t$ (when the displacement will be $s$).
# Alternatively, if we know an expression describing the displacement as a function of time, we can **differentiate** this expression w.r.t time in order to determine the velocity (and the constants of integration will vanish).
# 
# 
# We can see this in action in the examples shown in the lectures. In the first example we were told an expression describing the displacement $s=16t\left(4-t\right)$ and told to identify
# 
# 
# 1.   The time the object comes to rest.
# 2.   The initial velocity.
# 3.   The distance moved when the velocity is half its initial value.
# 4.   The acceleration when $t=4$.
# 
# Lets convert our expression to an equation in Python:

# In[2]:


t = sym.Symbol('t')
s = 16 * t * ( 4 - t )


# To evaluate the time when the velocity is zero, we need to recognise $v={\rm{d}}s/{\rm{d}}t$, so we require the time when the first derivative equals zero:

# In[3]:


time_whenatrest = sym.solve( sym.diff(s, t), t)
print("1. =", time_whenatrest)


# We can similarly solve for the initial velocity when $t=0$:

# In[4]:


v = sym.diff(s, t)
v_initial = v.evalf( subs={t:0} )
print("2. = ", v_initial)


# The distance when the velocity is half of the initial value

# In[5]:


v_halfinitial = 0.5 * v_initial
t_halfinitialv = sym.solve(v - v_halfinitial,t)
s_halfinitialv = s.evalf(subs={t:t_halfinitialv[0]})
print("3. = ", s_halfinitialv)


# The final part of the question requires calculation of velocity. Again from the lectures, we learned \begin{equation} \int_u^v ~{\rm{d}}v = \int_0^ta~{\rm{d}}t.\end{equation}
# This means that find an acceleration from a velocity, we differentiate velocity in time $a={\rm{d}}v/{\rm{d}}t$. If instead we know acceleration and require velocity, we must integrate.
# 
# For part four, we must differentiate our expression for velocity, which is the second derivative of the expression for position:
# 

# In[6]:


accel = sym.diff( s, t, t)
print("4. =", accel)


# Note that the acceleration here is a constant. This means that this system adheres to the equations of motion for constant acceleration derived in MAT101:
# \begin{eqnarray}
# v &=& u + at, \\
# v^2 &=& u^2 + 2as, \\
# s &=& ut + \frac{1}{2}at^2.
# \end{eqnarray}
# 
# Note that these expressions also differentiate or integrate to return each other, for example differentiating the expression for displacement returns:
# 

# In[7]:


u = sym.Symbol('u')
a = sym.Symbol('a')
s = u * t + 0.5 * a * t ** 2
print("s=", s) 
v = sym.diff(s,t)
print("ds/dt=", v)


# Python has rearranged the equations to that the highest power of t is first: this is because mathematicians usually arrange their equations in order of powers, with the highest power of the primary variable at the beginning.

# ## Variable acceleration
# 
# We've now seen an example of how Python can be used to tackle problems of integration and differentiation in the case where the acceleration is constant. Many problems we will encounter involving motion of bodies will contain forces (and hence accelerations) which vary in time.
# 
# We can use exactly the same tricks to try to solve these problems using *sympy*. To illustrate this, we'll work through a video example we saw in Lecture 2. NB: Python is great at solving symbolic maths problems but **you have to set up the problem correctly yourself first**: Python will try to solve any problem set before it, even if it is different to the problem you are actually trying to solve! You will see this in this example: we have to arrange the problem in a specific way for Python to solve.
# 
# Q: A body of mass $2~{\rm{kg}}$ is projected on a rough horizontal table which exerts a resistance given by $5v/2$ Newtons for velocity $v$. The initial velocity of the mass is $8~{\rm{ms}}^{-1}$. Find the time taken for the velocity to equal $4~{\rm{ms}}^{-1}$ and find the distance travelled in this time.
# 
# We'll start by defining variables and functions, and convert the force in the question into an acceleration through Newton's Second Law:

# In[8]:


v = sym.Symbol('v',real=True)
t = sym.Symbol('t',real=True)
v_ini = 8
mass = 2.0
F_res = -5.0 * v / 2.0
a = F_res/mass
print("a=",a)


# NB we have specifically restricted the type of variable $v$ and $t$ can be, so that Python won't look for Imaginary solutions later on in the calculation.

# From the lectures, we know that acceleration is the rate of change of velocity with time: 
# \begin{equation} a=\frac{{\rm{d}}v}{{\rm{d}}t}.
# \end{equation}
# This is a separable differential equation; multiplying both sides by ${\rm{d}}t$ and integrating yields:
# \begin{equation} 
# \int_{t=0}^{t=t}{a(t)}~{\rm{d}}t=\int_{v=u}^{v=v}{\rm{d}}v.
# \end{equation}
# However, our expression for acceleration depends on $v$, not $t$ (for now). So we must move any velocity dependence over to the right hand side, because this is where all the velocity terms are:
# \begin{eqnarray} 
# \int_{0}^{t}\frac{5v}{4}~{\rm{d}}t&=&\int_{u}^{v}{\rm{d}}v, \\
# \int_{0}^{t}\frac{5}{4}~{\rm{d}}t&=&\int_{u}^{v}\frac{{\rm{d}}v}{v}.
# \end{eqnarray}
# This is now an integral equation we can handle: lets make Python solve the left and right hand sides seperately:

# In[9]:


lhs = sym.integrate(a / v,(t, 0, t))
print("lhs: ",lhs)
rhs = sym.integrate(1.0 / v,(v, v_ini, v))
print("rhs: ",rhs)


# Having integrated each side separately, we can move one result over to the other side and ask Python to solve the equation "RHS-LHS=0":

# In[10]:


solution = sym.solve(rhs - lhs, v)
print("v(t):",solution)


# So our particular solution matches that found in the video example. Now our task is to see which value of $t$ makes this expression equal $4{\rm{ms}}^{-1}$:

# In[11]:


v_4 = 4.0
t_4 = sym.solve(solution[0] - v_4, t)
print("time taken for v=4m/s: ", t_4)


# The question also asks us to evaluate an expression for the displacement travelled in this time. We know that displacement and velocity are related via $v={\rm{d}s}/{\rm{d}}t$, so we can again multiply both sides by ${\rm{d}}t$ to create an integral equation:
# \begin{equation}
# \int_0^t{v}~{\rm{d}}t=\int_{s_0}^s {\rm{d}}s.
# \end{equation}
# The right hand side of this equation simply integrates to $s-s_0$, for initial displacement $s(t=0)=s_0$. If we assume this initial displacement is zero, then all we really want to know to answer the question is:
# to 
# 
# *   integrate the velocity once in time (yielding a general expression for the displacement as a function of time)
# *   evaluate that expression at the time we obtained earlier:
# 
# 

# In[12]:


s = sym.integrate(solution[0],(t, 0, t))
print("s(t)=",s)
s_t4 = s.evalf(subs={t:t_4[0]})
print("s(0.5545177)=",s_t4)


# The values we obtain for the time and displacement when the velocity reduces to $4{\rm{ms}}^{-1}$ agrees with the values seen in the example video performed by hand.

# ## Over To You
# 
# This worksheet illustrates how to use symbolic Python to relate displacement, velocity and acceleration through integration and differentiation. However, care has to be taken to set up the integral/differential equation correctly, depending on the objective.
# 
# Try this for yourself: identify one or two completed tutorial questions or lecture examples and see if you can repeat the processes seen above to complete the question using Python.

# In[ ]:




