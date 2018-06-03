import tensorflow as tf
import numpy as np

# Problem:
#
# Maintain W as an orthogonal matrix, while still allowing it to be learned
# 
# Solution:
# 
# Parametrize W by another matrix A in such a way that \phi(A) is always
# an orthogonal matrix. This implementation chooses W = C(A) @ D, where C is
# the Cayley Transform C(A) = (A+I)^-1 @ (A-I), and D is a fixed diagonal
# matrix. A does not necessarily remain skewsymmetric during learning, but by
# only storing the bottom half of the matrix, A is automatically mapped to a
# nearby skewsymmetric matrix via storage and restoration.


# A is skewsym, so W = Cayley(A) can't have eigenvalue -1. This is remedied by
# taking W = Cayley(A) @ D, where D is a diagonal matrix of 1s and -1s,
# the proportion of which is determined randomly from the range given below:

fraction_negeigs_bounds = [.33,.67]


# constructors for A and D, respectively

def _newA(shape):
  s = np.random.uniform(0, np.pi/2.0, size=int(np.floor(shape/2.0)))
  s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))
  z = np.zeros(s.size)
  if shape % 2 == 0:
      diag = np.hstack(zip(s, z))[:-1]
  else:
      diag = np.hstack(zip(s,z))
  A = np.diag(diag, k=-1)
  A = A - A.T
  return A.astype(np.float32)
  
def _newD(shape):
  lo = np.floor(shape * fraction_negeigs_bounds[0])
  hi = np.ceil (shape * fraction_negeigs_bounds[1])
  n_negeigs = np.random.randint(lo, hi)
  D = np.diag(np.concatenate([np.ones(shape - 
n_negeigs),-np.ones(n_negeigs)]))
  return D.astype(np.float32)


# the class to use; wraps some ugliness and minimizes use of
# O(n^3) operations, sacrificing a little storage.

class ortho_module:
  
  _from_A = {}
  
  def __init__(self, shape, name, no_bias=False):
    

    self.name = name
    
    M = np.zeros([shape, shape], dtype=np.float32)
    for i in range(shape):
      for j in range(i):
        M[i,j] = 1
    
    I = tf.eye(shape)
    
    self.M,self.I = M,I
    
    with tf.variable_scope(name):
      
      self.A = tf.get_variable('___A', initializer = _newA(shape))
      
      ortho_module._from_A[self.A] = self
      
      with tf.control_dependencies([self.A]):
        self.IpAi = tf.get_variable('IpAi', trainable=False,
             initializer=tf.matrix_solve_ls(I + self.A, I, fast=False))
                        
      with tf.control_dependencies([self.IpAi]):
        self.W = tf.get_variable('W', trainable=False,
             initializer=self.IpAi @ (I - self.A))
      
      self.D = tf.get_variable('D', trainable=False,
             initializer = _newD(shape))
      
      if not no_bias:
        self.b = tf.get_variable('b', [shape],
               initializer = tf.random_normal_initializer())
      else:
        self.b = None

    
  def __call__(self, X):
    '''keras-style call syntax; unused'''
    with tf.variable_scope(self.name):
      if self.b is not None:
        self.Z = tf.nn.relu(X @ self.W + self.b)
      
      else:
        self.Z = tf.nn.relu(X @ self.W)
      
    return self.Z

  def fromA(A): return ortho_module._from_A[A]
  
  def gradients(self, L):
    '''returns dLdA'''
    dLdW = tf.gradients(L, self.W)[0]
    V = tf.matrix_transpose(self.IpAi) @ dLdW @ (self.D + \
                                                 
tf.matrix_transpose(self.W))

    return [(V - tf.matrix_transpose(V), self.A)]
  
  def update(self):
    '''apply after change to A is made to resnyc parametrized objects'''
    A,I,M = self.A,self.I,self.M
    update_A = tf.assign(A, M * A)
    
    with tf.control_dependencies([update_A]):
      IpA = A - tf.matrix_transpose(A) + I
    
      update_IpAi = tf.assign(self.IpAi, 
tf.matrix_solve_ls(IpA,I,fast=False))
      
    with tf.control_dependencies([update_IpAi]):
      update_W = tf.assign(self.W, self.IpAi @ (A - I) @ self.D)
    
    with tf.control_dependencies([update_W]):
      with tf.variable_scope(self.name):
        update_all = tf.no_op(name='update')
    
    return update_all

