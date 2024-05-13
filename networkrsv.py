import matplotlib.pyplot as plt
import numpy as np
import os
import time



import tensorflow as tf

from scipy.linalg import circulant

class DynamicsFN:
  def __init__(self,fn,*args):
    self.fn = fn
    self.params = args[0]

  def __call__(self, y, t, x):
    return self.fn(y, t, x, self.params)


class NetworkRsv:
  def __init__(self, integrator=None):
    supported_integrators =   {"DormPrince"      : self.runge_kutta_dormand_prince,
                               "AdaptDormPrince" : self.adaptive_runge_kutta_dormand_prince,
                               "RK4"             : self.runge_kutta4}
    self.fn = None
    try:
      self.integrator = supported_integrators[integrator]
    except Exception as e:
      print("Integrator not in list - Falling back to default (Dormand Prince)")
      self.integrator = self.runge_kutta_dormand_prince
    
    self.y0 = tf.constant([0.0],dtype=tf.float64)
    self.t0 = tf.constant([0.0],dtype=tf.float64)
    self.x = tf.constant([0.0],dtype=tf.float64)

  def set_initial_cond(self, y0, t0):
    self.y0 = y0
    self.t0 = t0


  def set_dynamics(self, fn, fn_params):
    self.fn = DynamicsFN(fn, fn_params)

  def set_integrator(self, integrator):
    self.integrator = integrator


  def set_pace(self, t_max, step):
    self.t_max = t_max
    self.step = step

  @tf.function
  def runge_kutta_dormand_prince(self, f, t0, y0, t_end, h, x):
    y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = t.write(0,t0)
    y = y.write(0,y0)
    idx = tf.constant(0, dtype=tf.int32)

    def condition(t,y,x, idx):
      return tf.less(t.read(idx), t_end)
    def body(t,y,x, idx):
      y_temp = y.read(idx)
      t_temp = t.read(idx)
      x_temp = x[idx]

      k1 = h * f(t_temp, y_temp,x_temp)
      k2 = h * f(t_temp + h / 5, y_temp + k1 / 5,x_temp)
      k3 = h * f(t_temp + 3 * h / 10, y_temp + 3 * k1 / 40 + 9 * k2 / 40,x_temp)
      k4 = h * f(t_temp + 4 * h / 5, y_temp + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9,x_temp)
      k5 = h * f(t_temp + 8 * h / 9, y_temp + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729,x_temp)
      k6 = h * f(t_temp + h, y_temp + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656,x_temp)

      y_temp = y_temp + 35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84
      idx += 1
      y = y.write(idx, y_temp)

      t = t.write(idx, t_temp + h)
      tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, end='\r')
      return t, y, x, idx
    

    t, y, x, idx = tf.while_loop(condition, body, [t, y, x, idx],parallel_iterations=100)
    return y.stack(),t.stack()
  @tf.function
  def runge_kutta4(self, f, t0, y0, t_end, h, x):
    y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = t.write(0,t0)
    y = y.write(0,y0)
    idx = tf.constant(0, dtype=tf.int32)

    def condition(t,y,x, idx):
      return tf.less(t.read(idx), t_end)
    def body(t,y,x, idx):
      y_temp = y.read(idx)
      t_temp = t.read(idx)
      x_temp = x[idx]

      k1 = f(t_temp, y_temp, x_temp)
      k2 = f(t_temp + h/2., y_temp + h * k1 / 2., x_temp)
      k3 = f(t_temp + h/2., y_temp + h * k2 / 2., x_temp)
      k4 = f(t_temp + h   , y_temp + h * k3     , x_temp)

      y_temp = y_temp + h * (k1 + 2. * k2 + 2 * k3 + k4) / 6.
      idx += 1
      y = y.write(idx, y_temp)

      t = t.write(idx, t_temp + h)
      tf.print("Time step", tf.cast(t_temp, tf.int32), " out of ", t_end, end='\r')
      return t, y, x, idx
    

    t, y, x, idx = tf.while_loop(condition, body, [t, y, x, idx],parallel_iterations=100)
    return y.stack(),t.stack()
  @tf.function
  def adaptive_runge_kutta_dormand_prince(self, f, t0, y0, t_end, h, x, atol, rtol):
    y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = t.write(0,t0)
    y = y.write(0,y0)
    x_temp = tf.constant([0.0,0.0], dtype=tf.float64)
    idx = tf.constant(0, dtype=tf.int32)
    if tf.less(atol,h) or tf.less(rtol,h):
      tf.print("Warning! Either the relative or absolute error tolerance is smaller than the initial step, the integration might not proceed")
    def step_integ(f,t,y,h):
      k1 = h * f(t, y, x_temp)
      k2 = h * f(t + h / 5, y + k1 / 5, x_temp)
      k3 = h * f(t + 3 * h / 10, y + 3 * k1 / 40 + 9 * k2 / 40, x_temp)
      k4 = h * f(t + 4 * h / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9, x_temp)
      k5 = h * f(t + 8 * h / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729, x_temp)
      k6 = h * f(t + h, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656, x_temp)

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
      tf.print("Time step", t_temp, " out of ", t_end, "(iteration index: ", idx,")", end='\r')
      return t, y, idx,h, atol, rtol

    
    t, y, idx,_,_,_ = tf.while_loop(condition, body, [t, y, idx, h, atol, rtol],parallel_iterations=10)
    return y.stack(),t.stack()

  def fit(self, fn, y0, t0, t_max, x, step, fn_params = None, integrator_params = None):
    self.set_initial_cond(y0,t0)
    self.set_pace(t_max, step)
    self.set_dynamics(fn, fn_params) 
    self.integrator_params = integrator_params # Unused member for now - Passing integrator parameters directly in the run method.
    self.x = x

  def run(self, **kwargs):
    if self.fn == None:
      print("Function describing dynamics not found - Run `fit` before this function")
      return 1
    return self.integrator(self.fn, self.t0, self.y0, self.t_max, self.step,self.x, **kwargs)

