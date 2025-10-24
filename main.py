import matplotlib.pyplot as plt
import numpy as np
import os
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

from scipy.linalg import circulant

if os.environ.get("DISPLAY", "") == "" or "localhost" in os.environ.get("DISPLAY", ""):
    import matplotlib
    print("No display found. Using non-GUI Agg backend.")
    matplotlib.use("Agg")



from networkrsv import NetworkRsv
#try:
#  os.environ['CUDA_VISIBLE_DEVICES'] == ''
#  print("RUNNING ON CPU")
#except:
#    print("RUNNING ON GPU WITH CUDA")
try:
  print("DYNAMIC MEMORY ALLOCATION: ",os.environ['TF_FORCE_GPU_ALLOW_GROWTH'])
except:
  print("DYNAMIC MEMORY ALLOCATION: false")
def ER_adjacency_tensor(N, p, directed = True):
  rand = tf.random.uniform(shape = (N, N))
  adj  = tf.cast(tf.math.less(rand, p), tf.float64)
  if not directed:
      adj_ut = tf.linalg.band_part(adj, 0, -1)
      return adj_ut + tf.transpose(adj_ut)
  return adj

def ER_adjacency_sparse(N, p, directed = True):
    rand = tf.random.uniform(shape = (N, N))
    indices = tf.where(tf.math.less(rand,p))
    values = tf.ones(tf.shape(indices)[0], dtype = tf.float64)

    adj = tf.sparse.SparseTensor(indices, values, dense_shape = (N,N))
    adj = tf.sparse.reorder(adj)

    if not directed:
        adj_ut = tf.sparse.transpose(adj)
        adj = tf.sparse.add(adj, adj_ut)
    return adj

def physical_laplacian(A, normalised = True):
  degree_sequence = tf.reduce_sum(A, axis = 1)
  D = tf.linalg.diag(degree_sequence)
  L = D - A
  if normalised:
      Dinv = tf.linalg.diag([1./d if d > 0. else 0. for d in degree_sequence])
      Dinv =tf.cast(Dinv,dtype=tf.float64)
      L = tf.matmul(Dinv, L)
  return L

def physical_laplacian_sparse(A, normalised = True):
    degree_sequence = tf.sparse.reduce_sum(A, axis = 1)
    N = tf.shape(degree_sequence)[0]

    diag_idxs = tf.stack([tf.range(N), tf.range(N)], axis = 1)
    diag_vals = tf.cast(degree_sequence, tf.float64)
    D = tf.sparse.SparseTensor(tf.cast(diag_idxs,tf.int64), diag_vals, (N,N))

    L = tf.sparse.add(D, tf.sparse.map_values(tf.negative, A)) # L = D + (-A)

    if normalised:
        Dinv_val = tf.where(degree_sequence > 0, 1.0 / degree_sequence, 0.0)
        Dinv = tf.sparse.SparseTensor(tf.cast(diag_idxs,tf.int64), Dinv_val, (N,N))
        #L = tf.sparse.matmul(Dinv, L)
        L = tf.sparse.sparse_dense_matmul(Dinv, tf.sparse.to_dense(L))
        L = tf.sparse.from_dense(L)
    
    return tf.sparse.reorder(L)


@tf.function
def ode_fn(t, y, x, params):
# params = \eps, a, J (coupling strength), L (coupling matrix), I (external drivers)
  u,v = tf.split(y,2)
  ux,vx = tf.split(x,2)
  eps, a, J, L, I = params
  #tf.print(ux)
  
  return tf.reshape(tf.stack([(u - tf.pow(u,3)/3 - v - J*tf.linalg.matvec(L,u,a_is_sparse = True) +I*ux)/eps,
                               u + a],axis=0),[-1])

@tf.function
def sparse_ode_fn(t, y, x, params):
    u,v = tf.split(y,2)
    ux,vx = tf.split(x,2)
    eps, a, J, L, I = params
    
    Lu = tf.sparse.sparse_dense_matmul(L, tf.expand_dims(J*u,1))
    Lu = tf.squeeze(Lu, axis=1)

    du = (u - tf.pow(u,3)/3 - v - Lu +I*ux) / eps
    dv = u + a

    return tf.reshape(tf.stack([du, dv], axis = 0),[-1])

solver = NetworkRsv("stochRK4")
N=200
t_init = tf.constant(0., dtype=tf.float64)
t_max = tf.constant(100., dtype=tf.float64)
step = tf.constant(0.001, dtype=tf.float64)
expected_steps = 1 + int(t_max/step)

T = 1.5e-2/1e-2
a = 1.3
fp = (-a, a**3/3 - a)
# y_init = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype = tf.float64)
# y_init = tf.random.uniform([1000], dtype = tf.float64)
y_init = tf.random.uniform([2*N], dtype = tf.float64)
#    x.write(i,tf.zeros([2*N],dtype=tf.float64))
x = []
for i in range(int(2*t_max/step)+1):
  x.append(tf.zeros([2*N],dtype=tf.float64))
  if i % 1000 == 0:
    x[i] = tf.random.normal([N*2], dtype=tf.float64)
  #x.append(tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64))
  
tf.print(tf.shape(x)[0])
tf.print(int(2*t_max/step)+1)
# pulse_step = tf.random.uniform([],maxval=int(t_max/step)+1,dtype=tf.int32)
# x[pulse_step] = tf.constant([fp[0] + 5 for i in range(N)] + [fp[1]] * N, dtype=tf.float64)
# x[pulse_step+ 1] = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64)
# x[pulse_step+ 2] = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64)
# x[pulse_step+ 3] = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64)
# x[pulse_step+ 4] = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64)
# x[pulse_step] = tf.random.normal([N*2], dtype=tf.float64)
# print(x[pulse_step])
# print(pulse_step)
x = tf.convert_to_tensor(x, dtype=tf.float64)
#x = tf.zeros([int(t_max/step)+1,2*N], dtype = tf.float64)
tf.print(x.shape)
atol = tf.constant(1e-3,dtype=tf.float64)
rtol = tf.constant(1e-3,dtype=tf.float64)
tf.print("Initial state ",y_init)

#A = ER_adjacency_tensor(N, 0.01, True)#tf.convert_to_tensor(circulant([0 for i in range(N - 1)] + [1]).T, dtype = tf.float64)
#L = physical_laplacian(A) 

A = ER_adjacency_sparse(N, 0.01, True)
L = physical_laplacian_sparse(A)

I = tf.constant(0., dtype = tf.float64)


params = tf.tuple([0.01, a, 0.1, L, I]) 

solver.fit(sparse_ode_fn,y_init,t_init,t_max,x, step, params)

print("Starting integration")
start_time= time.time()
results = solver.run(T = T)

print("Integrated ", len(results[1]), " steps in ", time.time() - start_time, " s")


tf.print("Results", results)
n_plots = len(results[0][0])
if not os.path.exists("plots"):
    os.makedirs("plots")
for res in range(min(int((n_plots*0.01)/2),20)):   #Plotto solo l'1% delle coppie X,Y ma mai più di 20
  fig, ax = plt.subplots()
  ax.plot(results[1],results[0][:,res],label="x")
  ax.plot(results[1],results[0][:,int(res+n_plots/2)],label="y")
  ax.set_xlabel("t")

  ax.legend(loc="upper right")
  fig.savefig("plots/evoVSt_" + str(res) + ".png")
  
  fig, ax = plt.subplots()
  print(res, int(res+n_plots/2))
  sc = ax.scatter(results[0][:,res],results[0][:,int(res+n_plots/2)],c=results[1])
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  fig.colorbar(sc, label= "t")
  fig.savefig("plots/XYevo_" + str(res) + ".png")
  plt.clf()

# fig_heat, ax_heat = plt.subplots()
# 
# ax_heat.imshow(tf.transpose(results[0][:, :N]))
# ax_heat.set_xlabel("integration step")
# ax_heat.set_ylabel("node index")
# ax_heat.set_xlim(right = len(results[1])//20)
# plt.show()
