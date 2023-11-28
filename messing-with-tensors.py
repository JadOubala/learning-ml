#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch;


# In[11]:


x = torch.arange(12, dtype=torch.float32)


# In[12]:


x


# In[13]:


x.numel()


# In[14]:


x.shape


# In[15]:


X = x.reshape(3, 4)


# In[17]:


X


# Note that specifying every shape component to reshape is redundant. Because we already know our tensor's size, we can work out one component of the shape given the rest. For example, given a tensor of size  𝑛and target shape ( ℎ,  𝑤), we know that  𝑤=𝑛/ℎ.
# To automatically infer one component of the shape, we can place a -1 for the shape component that should be inferred automatically. In our case, instead of calling x.reshape(3, 4), we could have equivalently called x.reshape(-1, 4) or x.reshape(3, -1).
# Practitioners often need to work with tensors initialized to contain all zeros or ones. [We can construct a tensor with all elements set to zero] (or one) and a shape of (2, 3, 4) via the zeros function.

# In[25]:


x.reshape(-1, 4)


# In[26]:


x.reshape(3, -1)


# In[27]:


torch.zeros((2, 3, 4))


# In[28]:


torch.ones((2, 3, 4))


# In[31]:


torch.randn(3, 4)


# In[32]:


torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# ## Indexing and Slicing
# 
# As with  Python lists,
# we can access tensor elements 
# by indexing (starting with 0).
# To access an element based on its position
# relative to the end of the list,
# we can use negative indexing.
# Finally, we can access whole ranges of indices 
# via slicing (e.g., `X[start:stop]`), 
# where the returned value includes 
# the first index (`start`) *but not the last* (`stop`).
# Finally, when only one index (or slice)
# is specified for a $k^\mathrm{th}$ order tensor,
# it is applied along axis 0.
# Thus, in the following code,
# [**`[-1]` selects the last row and `[1:3]`
# selects the second and third rows**].

# In[37]:


X[-1], X[1:3]


# In[38]:


X


# In[41]:


X[1, 2] = 17
X


# In[42]:


# Tensor accessing is pretty straight-forward, recall zero-based indexing


# In[43]:


# If we want [to assign multiple elements the same value, we apply the indexing on the left-hand side of the assignment operation.]


# In[45]:


X[:2, :] = 12
X


# In[ ]:


# Operations with tensors:
# E^(member), unary scalar operator


# In[46]:


torch.exp(x)


# Likewise, we denote *binary* scalar operators,
# which map pairs of real numbers
# to a (single) real number
# via the signature 
# $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
# Given any two vectors $\mathbf{u}$ 
# and $\mathbf{v}$ *of the same shape*,
# and a binary operator $f$, we can produce a vector
# $\mathbf{c} = F(\mathbf{u},\mathbf{v})$
# by setting $c_i \gets f(u_i, v_i)$ for all $i$,
# where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
# of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
# Here, we produced the vector-valued
# $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
# by *lifting* the scalar function
# to an elementwise vector operation.
# The common standard arithmetic operators
# for addition (`+`), subtraction (`-`), 
# multiplication (`*`), division (`/`), 
# and exponentiation (`**`)
# have all been *lifted* to elementwise operations
# for identically-shaped tensors of arbitrary shape.

# In[51]:


x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y


# In addition to elementwise computations,
# we can also perform linear algebra operations,
# such as dot products and matrix multiplications.
# 
# We can also [***concatenate* multiple tensors together,**]
# stacking them end-to-end to form a larger tensor.
# We just need to provide a list of tensors
# and tell the system along which axis to concatenate.
# The example below shows what happens when we concatenate
# two matrices along rows (axis 0)
# vs. columns (axis 1).
# We can see that the first output's axis-0 length ($6$)
# is the sum of the two input tensors' axis-0 lengths ($3 + 3$);
# while the second output's axis-1 length ($8$)
# is the sum of the two input tensors' axis-1 lengths ($4 + 4$).
# 

# In[57]:


X = torch.arange(12, dtype = torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X,Y), dim=1)


# What if we want to [**construct a binary tensor via *logical statements*?**]
# Take `X == Y` as an example.
# For each position `i, j`, if `X[i, j]` and `Y[i, j]` are equal, 
# then the corresponding entry in the result takes value `1`,
# otherwise it takes value `0`.
# 

# In[251]:


print(X == Y)
print(X > Y)
print(X < Y)


# If we wanted to apply the logic to say, make all the corresponding values 66 and
# non-matching 42, we could do this:

# In[64]:


L = torch.tensor([[1, 2], [3, 4]])
M = torch.tensor([[1, 0], [3, 7]])
comparison = X == Y

result_tensor = torch.where(comparison, torch.full_like(X, 66), torch.full_like(X, 42))

print(result_tensor)


# Or...

# In[66]:


L = torch.tensor([[1, 2], [3, 4]])
M = torch.tensor([[1, 0], [3, 7]])

# Initialize a result tensor with the same shape
result_tensor = torch.empty_like(L)

# Loop over each element in the tensors
for i in range(L.shape[0]):      # loop over rows
    for j in range(L.shape[1]):  # loop over columns
        # Use an if-else conditional to check for equality and assign values
        if L[i, j] == M[i, j]:
            result_tensor[i, j] = 66
        else:
            result_tensor[i, j] = 42

print(result_tensor)


# [**Summing all the elements in the tensor**] yields a tensor with only one element.
# 

# In[61]:


X.sum()


# In[ ]:


import torch
import numpy as num

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")

x = torch.ones(5, requires_grad=True)
# tells pytorch it will need to calculate gradience for this tensor later during optimization
print(x)
    
a = numpy.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)


# ## Broadcasting
# 
# By now, I know how to perform 
# elementwise binary operations
# on two tensors of the same shape. 
# Under certain conditions,
# even when shapes differ, 
# we can still [**perform elementwise binary operations
# by invoking the *broadcasting mechanism*.**]
# Broadcasting works according to 
# the following two-step procedure:
# (i) expand one or both arrays
# by copying elements along axes with length 1
# so that after this transformation,
# the two tensors have the same shape;
# (ii) perform an elementwise operation
# on the resulting arrays.
# 

# In[215]:


a = torch.arange(3)
print(a)

b = torch.arange(2)
print(b)

print("")

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("a is ") 
print(a)
print("")
print("b is ")
print(b)


# In[239]:


a + b


# ## Saving Memory
# 
# [**Running operations can cause new memory to be
# allocated to host results.**]
# For example, if we write `Y = X + Y`,
# we dereference the tensor that `Y` used to point to
# and instead point `Y` at the newly allocated memory.
# We can demonstrate this issue with Python's `id()` function,
# which gives us the exact address 
# of the referenced object in memory.
# Note that after we run `Y = Y + X`,
# `id(Y)` points to a different location.
# That is because Python first evaluates `Y + X`,
# allocating new memory for the result 
# and then points `Y` to this new location in memory.
# 

# In[240]:


# Oh boy, a reminder to pointer arithmetic (pain and suffering)


# In[241]:


before = id(Y)
Y = Y + X
id(Y) == before


# This might be undesirable for two reasons.
# First, we do not want to run around
# allocating memory unnecessarily all the time.
# In machine learning, we often have
# hundreds of megabytes of parameters
# and update all of them multiple times per second.
# Whenever possible, we want to perform these updates *in place*.
# Second, we might point at the 
# same parameters from multiple variables.
# If we do not update in place, 
# we must be careful to update all of these references,
# lest we spring a memory leak 
# or inadvertently refer to stale parameters.
# 
# Fortunately, (**performing in-place operations**) is easy.
# We can assign the result of an operation
# to a previously allocated array `Y`
# by using slice notation: `Y[:] = <expression>`.
# To illustrate this concept, 
# we overwrite the values of tensor `Z`,
# after initializing it, using `zeros_like`,
# to have the same shape as `Y`.
# 

# In[243]:


Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))


# [**If the value of `X` is not reused in subsequent computations,
# we can also use `X[:] = X + Y` or `X += Y`
# to reduce the memory overhead of the operation. 
# This'll save some memory instead of creating another Z variable**]
# 

# In[244]:


before = id(X)
X += Y
id(X) == before


# ## Conversion to Other Python Objects
# 
# [**Converting to a NumPy tensor (`ndarray`)**], or vice versa, is easy.
# The torch Tensor and numpy array 
# will share their underlying memory, 
# and changing one through an in-place operation 
# will also change the other.
# 

# In[246]:


A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)


# To (**convert a size-1 tensor to a Python scalar**),
# we can invoke the `item` function or Python's built-in functions. 

# In[249]:


a = torch.tensor([3.5])
a, a.item(), float(a), int(a) 


# ### Summary 
# The tensor class is the main interface for storing and manipulating data in deep learning libraries. Tensors provide a variety of functionalities including construction routines; indexing and slicing; basic mathematics operations; broadcasting; memory-efficient assignment; and conversion to and from other Python objects.

# In[ ]:




