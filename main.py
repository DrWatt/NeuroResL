import matplotlib.pyplot as plt
import numpy as np
import os
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

from scipy.linalg import circulant


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

def physical_laplacian(A, normalised = True):
  degree_sequence = tf.reduce_sum(A, axis = 1)
  Dinv = tf.linalg.diag([1./d if d > 0. else 0. for d in degree_sequence])
  Dinv =tf.cast(Dinv,dtype=tf.float64)
  D = tf.linalg.diag(degree_sequence)
  L = D - A
  if normalised:
      L = tf.matmul(Dinv, L)
  return tf.sparse.from_dense(L)

@tf.function
def ode_fn(t, y, x, params):
# params = \eps, a, J (coupling strength), L (coupling matrix), I (external drivers)
  u,v = tf.split(y,2)
  ux,vx = tf.split(x,2)
  eps, a, J, L, I = params
  #tf.print(ux)
  du = (u - tf.pow(u,3)/3 - v)/eps
  dv = u + a
  u_ = tf.expand_dims(u,1)
  mtml = tf.sparse.sparse_dense_matmul(L,u_)
  inp= -(J*tf.squeeze(mtml) +I*ux)/eps
  return tf.reshape(tf.stack([du + inp, dv],axis=0),[-1])

solver = NetworkRsv("RK4")
N=1500
t_init = tf.constant(0., dtype=tf.float64)
t_max = tf.constant(10., dtype=tf.float64)
step = tf.constant(0.001, dtype=tf.float64)
expected_steps = 1 + int(t_max/step)

a  = 1.3
fp = (-a, a**3/3 - a)
# y_init = tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype = tf.float64)
# y_init = tf.random.uniform([1000], dtype = tf.float64)
y_init = tf.random.uniform([2*N], dtype = tf.float64)
#    x.write(i,tf.zeros([2*N],dtype=tf.float64))
x = []
for i in range(int(t_max/step)+1):
  x.append(tf.zeros([2*N],dtype=tf.float64))
  if i % 100 == 0:
    x[i] = tf.random.normal([N*2], dtype=tf.float64)
  # x.append(tf.constant([fp[0] + np.random.normal() for i in range(N)] + [fp[1]] * N, dtype=tf.float64))
  
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

A = ER_adjacency_tensor(N, 0.01, True)#tf.convert_to_tensor(circulant([0 for i in range(N - 1)] + [1]).T, dtype = tf.float64)
L = physical_laplacian(A) 



I = tf.constant(10., dtype = tf.float64)


params = tf.tuple([0.01, a, 0., L, I]) 


solver.fit(ode_fn,y_init,t_init,t_max,x, step, params)

print("Starting integration")
start_time= time.time()
results = solver.run()

results[1] = results[1].close().stack().numpy()
results[0] = results[0].close().stack().numpy()

print("Integrated ", len(results[1]), " steps in ", time.time() - start_time, " s")


#tf.print("Results", results)
n_plots = len(results[0][0])
if not os.path.exists("plots"):
    os.makedirs("plots")
for res in range(int((n_plots*0.01)/2)):  #Plotto solo l'1% delle coppie X,Y
  fig, ax = plt.subplots()
  ax.plot(results[1],results[0][:,res],label="x")
  ax.plot(results[1],results[0][:,int(res+n_plots/2)],label="y")
  ax.set_xlabel("t")

  ax.legend(loc="upper right")
  fig.savefig("plots/evoVSt_" + str(res) + ".png")
  
  fig, ax = plt.subplots()
#  print(res, int(res+n_plots/2))
  ax.scatter(results[0][:,res],results[0][:,int(res+n_plots/2)],c=tf.linalg.normalize(results[1])[0].numpy())
  ax.set_xlabel("x")
  ax.set_ylabel("y")

  fig.savefig("plots/XYevo_" + str(res) + ".png")
  plt.clf()

# fig_heat, ax_heat = plt.subplots()
# 
# ax_heat.imshow(tf.transpose(results[0][:, :N]))
# ax_heat.set_xlabel("integration step")
# ax_heat.set_ylabel("node index")
# ax_heat.set_xlim(right = len(results[1])//20)
# plt.show()
