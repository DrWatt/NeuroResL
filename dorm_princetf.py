import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import os
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


@tf.function
def ode_fn(t, y, A):
  return tf.linalg.matvec(A, y)



@tf.function
def runge_kutta_dormand_prince(f, t0, y0, t_end, h, A):
  # int((t_max - t0) / h)
  y = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
  t = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
  t = t.write(0,t0)
  y = y.write(0,y0)
  idx = tf.constant(0, dtype=tf.int32)
  e = tf.constant(5, dtype=tf.int32)
  t.read(idx)
  y.read(idx)
  def condition(t,y,idx):
    return tf.less(t.read(idx), t_end)
  def body(t,y,idx):
    y_temp = y.read(idx)
    t_temp = t.read(idx)
    k1 = h * f(t_temp, y_temp, A)
    k2 = h * f(t_temp + h / 5, y_temp + k1 / 5, A)
    k3 = h * f(t_temp + 3 * h / 10, y_temp + 3 * k1 / 40 + 9 * k2 / 40, A)
    k4 = h * f(t_temp + 4 * h / 5, y_temp + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9, A)
    k5 = h * f(t_temp + 8 * h / 9, y_temp + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729, A)
    k6 = h * f(t_temp + h, y_temp + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656, A)

    y_temp = y_temp + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84

    idx += 1
    y = y.write(idx, y_temp)

    t = t.write(idx, t_temp + h)
    tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, end='\r')
    return t, y, idx
  

  t, y, idx = tf.while_loop(condition, body, [t, y, idx],parallel_iterations=10)
  return y.stack(),t.stack()

@tf.function
def adaptive_runge_kutta_dormand_prince(f, t0, y0, t_end, h, A, atol, rtol):
  # int((t_max - t0) / h)
  y = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
  t = tf.TensorArray(dtype=tf.float32, size = 0, dynamic_size=True, clear_after_read=False)
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
    h_next = h * tf.minimum(5.0, tf.maximum(0.2, 0.9 * scale * tf.pow(error, -0.2)))
    # Accept the step if error is within tolerance
    if tf.less(error, atol):
      idx += 1
      t_temp = t_temp + h
      y_temp = y1
      y = y.write(idx, y_temp)
      t = t.write(idx, t_temp)

    else:
      h = h_next
    tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, end='\r')
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

  

t_init = tf.constant(0., dtype=tf.float32)
t_max = tf.constant(100., dtype=tf.float32)
step = tf.constant(0.01, dtype=tf.float32)
y_init = tf.constant([1., 1.], dtype=tf.float32)
atol = tf.constant(1e-1,dtype=tf.float32)
rtol = tf.constant(1e-1,dtype=tf.float32)
print(y_init)
# A = tf.constant([[0., -1., 0, 0], [1., 0., 0, 0],[0., -0.5, 0.5, 0], [0.5, 0.,0., -0.5]], dtype=tf.float32)
A = tf.constant([[0., -1], [1., 0.]], dtype=tf.float32)



print("Starting integration")
start_time= time.time()
results = solver(ode_fn, t_init, y_init, t_max, step, A)
print("Integrated ", len(results[1]), " steps in ", time.time() - start_time, " s")

tf.print("Results", results)

for res in range(len(results[0][0])):
  plt.plot(results[1],results[0][:,res])
  plt.show()

