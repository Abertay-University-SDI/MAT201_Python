#!/usr/bin/env python
# coding: utf-8

# # Fluid Resistance and Terminal speed
# Using Python's symbolic computation library, we will see how resistive forces affect motion.
# 
# If you have not already done so, please check out the MAT201 Python introduction notebook to better understand the content shown here.

# In Lecture 3 of our Dynamics component of MAT201, we explored ideas of changing the balance of forces acting upon an object, particularly in the case where a resistive force is applied proportional to the velocity of the object. 
# 
# We'll explore some of the same ideas and use Python to solve these equations for us.

# Let's go ahead and load in the libraries we need: in this case, we'll focus solely on sympy:
# 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sympy as sym


# ## Resistive forces
# 
# We have already considered general cases where an object may be subject to a resistive force. To recap, we saw that an object subject to a resistive force $R(v)$ would therefore be subject to Newton's Second Law as:
# \begin{equation}
#  m\frac{{\rm d}v}{{\rm d}t}=-R(v),
# \end{equation}
# which can be separated and integrated to determine expresions for elapsed time $t$ as 
# \begin{equation}
# t=-m\int_{v_0}^{v}\frac{{\rm d}v}{R(v)},
# \end{equation}
# or indeed an expression for the object displacement $s$ as
# \begin{equation}
# s=s_0+\int_{t=0}^{t=t}v{\rm d}t.
# \end{equation}

# In[ ]:





# In[2]:


t = sym.Symbol('t')
s = 16 * t * ( 4 - t )


# Now we can tackle Part 1.
# The object will be at rest when the velocity $=0$. To evaluate the time when this occurs, we first need to see how the velocity changes in time, by differentiating $s(t)$: $v={\rm{d}}s/{\rm{d}}t$. We then need to solve when this equals zero, for time $t$. We can ask Python to differentiate our expression for $s$ (giving us velocity) *and* solve when $v=0$, all in one line:

# In[3]:


time_whenatrest = sym.solve( sym.diff(s, t), t)
print("1. =", time_whenatrest)


# Part 2 requires the initial velocity. The initial velocity occurs when $t=0$, so we can ask Python to substitute the expression "t=0" into the velocity and solve:

# In[4]:


v = sym.diff(s, t)
v_initial = v.evalf( subs={t:0} )
print("2. = ", v_initial)


# Part 3 requires us to find the distance when the velocity has decreased to approximately half of its original value. This one is more complicated to set up, as we first have to calculate how long it will take for the velocity to drop by 0.5, and then substitute that value of $t$ into the expression we have for distance:

# In[5]:


v_halfinitial = 0.5 * v_initial
t_halfinitialv = sym.solve(v - v_halfinitial,t)
s_halfinitialv = s.evalf(subs={t:t_halfinitialv[0]})
print("3. = ", s_halfinitialv)


# Part 4 requires an acceleration, something we haven't yet mentioned. Again from the lectures, we learned \begin{equation} \int_u^v ~{\rm{d}}v = \int_0^ta~{\rm{d}}t.\end{equation}
# This means that, to find an acceleration from a velocity, we differentiate velocity in time $a={\rm{d}}v/{\rm{d}}t$. If instead we know acceleration and require velocity, we must integrate.
# 
# So to solve Part 4, we will differentiate the velocity in time. But our velocity expression is *already* a differential: we can calculate the acceleration by differentiating the displacement *twice*:
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
# You should be able to show that hese expressions also differentiate or integrate to return each other, for example differentiating the expression for displacement returns:
# 

# In[7]:


u = sym.Symbol('u')
a = sym.Symbol('a')
s = u * t + 0.5 * a * t ** 2
print("s=", s) 
v = sym.diff(s,t)
print("ds/dt=", v)


# Python akways rearranges the equations so that the highest power of the variable (in this case $t$) is first: this is because mathematicians usually arrange their equations in order of powers, with the highest power of the primary variable at the beginning.

# ## Variable acceleration
# 
# We've now seen an example of how Python can be used to tackle problems of integration and differentiation in the case where the acceleration is constant. Many problems we will encounter involving motion of bodies will contain forces (and hence accelerations) which vary in time.
# 
# We can use exactly the same tricks to try to solve these problems using *sympy*. To illustrate this, we'll work through a video example we saw in Lecture 2. NB: Python is great at solving symbolic maths problems but **you have to set up the problem correctly yourself first**: Python will try to solve any problem set before it, even if it is different to the problem you are actually trying to solve! You will see this in this example: we have to arrange the problem in a specific way in order that Python will solve it correctly.
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




