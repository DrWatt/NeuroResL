import matplotlib.pyplot as plt
import numpy as np
import os
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf


@tf.function
def ode_fn(t, y, params):
  a,b = tf.split(y,2)
  #tf.print("result of f",tf.reshape(tf.stack([(a - tf.pow(a,3)/3 - b + 0.1)/0.1,a + 0.1],axis=0),[-1]))
  return tf.reshape(tf.stack([(a - tf.pow(a,3)/3 - b + params[2])/params[0],a + params[1]],axis=0),[-1])
  #tf.stack([(y[0] - tf.pow(y[0],3)/3 - y[1] + 0.1)/0.1,y[0] + 0.1])
  #return tf.linalg.matvec(A, y)



@tf.function
def runge_kutta_dormand_prince(f, t0, y0, t_end, h, params):
  # int((t_max - t0) / h)
  y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
  t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
  t = t.write(0,t0)
  y = y.write(0,y0)
  idx = tf.constant(0, dtype=tf.int32)
  t.read(idx)
  y.read(idx)
  def condition(t,y,idx):
    return tf.less(t.read(idx), t_end)
  def body(t,y,idx):
    y_temp = y.read(idx)
    t_temp = t.read(idx)
    k1 = h * f(t_temp, y_temp, params)
    k2 = h * f(t_temp + h / 5, y_temp + k1 / 5, params)
    k3 = h * f(t_temp + 3 * h / 10, y_temp + 3 * k1 / 40 + 9 * k2 / 40, params)
    k4 = h * f(t_temp + 4 * h / 5, y_temp + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9, params)
    k5 = h * f(t_temp + 8 * h / 9, y_temp + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729, params)
    k6 = h * f(t_temp + h, y_temp + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656, params)

    y_temp = y_temp + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84
    # tf.print("y_temp", y_temp, "idx",idx)
    idx += 1
    y = y.write(idx, y_temp)

    t = t.write(idx, t_temp + h)
    tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, end='\r')
    return t, y, idx
  

  t, y, idx = tf.while_loop(condition, body, [t, y, idx],parallel_iterations=100)
  return y.stack(),t.stack()

@tf.function
def adaptive_runge_kutta_dormand_prince(f, t0, y0, t_end, h, A, atol, rtol):
  # int((t_max - t0) / h)
  y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
  t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
  t = t.write(0,t0)
  y = y.write(0,y0)
  idx = tf.constant(0, dtype=tf.int32)
  def step_integ(f,t,y,h):
    k1 = h * f(t, y, A)
    k2 = h * f(t + h / 5, y + k1 / 5, A)
    k3 = h * f(t + 3 * h / 10, y + 3 * k1 / 40 + 9 * k2 / 40, A)
    k4 = h * f(t + 4 * h / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9, A)
    k5 = h * f(t + 8 * h / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729, A)
    k6 = h * f(t + h, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656, A)

    y_next = y + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84

    return y_next

  def condition(t,y,idx,h, atol, rtol):
    return tf.less(t.read(idx), t_end)
  def body(t,y,idx,h, atol, rtol):
    y_temp = y.read(idx)
    t_temp = t.read(idx)

    y1 = step_integ(f, t_temp, y_temp, h)
    y2 = step_integ(f, t_temp + h / 2, y1, h / 2)
    # Error estimation
    error = tf.reduce_max(tf.abs(y2 - y1))
    
    # Adaptive step size control
    scale = tf.pow((atol + rtol * tf.maximum(tf.reduce_max(tf.abs(y1)), tf.reduce_max(tf.abs(y2)))), 0.25)
    h_next = h * tf.minimum(tf.cast(5.0,dtype=tf.float64), tf.maximum(tf.cast(0.2,dtype=tf.float64), 0.9 * scale * tf.pow(error, -0.2)))
    # Accept the step if error is within tolerance
    if tf.less(error, atol):
      idx += 1
      t_temp = t_temp + h
      y_temp = y1
      y = y.write(idx, y_temp)
      t = t.write(idx, t_temp)

    else:
      h = h_next
    tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, "(iteration index: ", idx,")", end='\r')
    return t, y, idx,h, atol, rtol

  
  t, y, idx,_,_,_ = tf.while_loop(condition, body, [t, y, idx,h, atol, rtol],parallel_iterations=10)
  return y.stack(),t.stack()


@tf.function
def solver(fn, t_init, y_init, t_max, step, A, atol=None, rtol=None):
  # return tfp.math.ode.DormandPrince().solve(fn, t_init, y_init, solution_times=np.linspace(1,1000,10))
  if atol == rtol == None:
    return runge_kutta_dormand_prince(fn, t_init, y_init, t_max, step, A)
  else:
    return adaptive_runge_kutta_dormand_prince(fn, t_init, y_init, t_max, step, A, atol,rtol)

  

t_init = tf.constant(0., dtype=tf.float64)
t_max = tf.constant(10., dtype=tf.float64)
step = tf.constant(0.0001, dtype=tf.float64)
# y_init = tf.constant([1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.], dtype=tf.float64)
y_init = tf.random.normal([1000], dtype = tf.float64)
atol = tf.constant(1e-4,dtype=tf.float64)
rtol = tf.constant(1e-4,dtype=tf.float64)
tf.print("Initial state ",y_init)
# A = tf.constant([[0., -1., 0, 0], [1., 0., 0, 0],[0., -0.5, 0.5, 0], [0.5, 0.,0., -0.5]], dtype=tf.float64)
params = tf.Variable([0.01, 1.3,0.1], dtype=tf.float64) # Params: eta, a, driver



print("Starting integration")
start_time= time.time()
results = solver(ode_fn, t_init, y_init, t_max, step, params)
#results = ode_fn(t_init,y_init,A)
print("Integrated ", len(results[1]), " steps in ", time.time() - start_time, " s")

tf.print("Results", results)
n_plots = len(results[0][0])
for res in range(int(n_plots*0.01)):  #Plotto solo l'1% delle coppie X,Y
  fig, ax = plt.subplots()
  ax.plot(results[1],results[0][:,res])
  if res < n_plots/2:
    ax.set_xlabel("X(" + str(res) +")")
    # plt.show()
    plt.savefig("plots/evo_" + str(res) + "_X.png")
    plt.clf()
  else:
    ax.set_xlabel("Y(" + str(int(res-n_plots/2)) +")")
    # plt.show()
    plt.savefig("plots/evo_" + str(int(res-n_plots/2)) + "_Y.png")
    plt.clf()
  if res < n_plots/2:
    print(res, int(res+n_plots/2))
    plt.plot(results[0][:,res],results[0][:,int(res+n_plots/2)])
    # plt.show()
    plt.savefig("plots/XYevo_" + str(res) + ".png")
  
# plt.plot(results[0][:,0],results[0][:,3])
# plt.show()