import matplotlib.pyplot as plt
import numpy as np
import os
import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

from scipy.linalg import circulant

class DynamicsFN:
  def __init__(self,fn,*args):
    self.fn = fn
    self.params = args[0]

  def __call__(self, y, t):
    return self.fn(y,t,self.params)


class NetworkRsv:
  def __init__(self, integrator=None):
    supported_integrators =   {"DormPrince" : self.runge_kutta_dormand_prince,"AdaptDormPrince" : self.adaptive_runge_kutta_dormand_prince  }
    self.fn = None
    try:
      self.integrator = supported_integrators[integrator]
    except Exception as e:
      print("Integrator not in list - Falling back to default (Dormand Prince)")
      self.integrator = self.runge_kutta_dormand_prince
    
    self.y0 = tf.constant([0.0],dtype=tf.float64)
    self.t0 = tf.constant([0.0],dtype=tf.float64)

  # @tf.function
  def set_initial_cond(self, y0, t0):
    self.y0 = y0
    self.t0 = t0

  # @tf.function
  def set_dynamics(self, fn, fn_params):
    self.fn = DynamicsFN(fn, fn_params)
  # @tf.function
  def set_integrator(self, integrator):
    self.integrator = integrator

  # @tf.function
  def set_pace(self, t_max, step):
    self.t_max = t_max
    self.step = step

  @tf.function
  def runge_kutta_dormand_prince(self, f, t0, y0, t_end, h):
    # int((t_max - t0) / h)
    y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = t.write(0,t0)
    y = y.write(0,y0)
    idx = tf.constant(0, dtype=tf.int32)
    t.read(idx)
    y.read(idx)
    a = 1.3
    fp = (-a, a**3/3 - a)
    # pulse = tf.constant([fp[0] + 5] + [fp[0]]*(500-1) + [fp[1]] * 500, dtype = tf.float64)
    pulse = tf.concat([tf.random.normal([500], dtype=tf.float64),tf.random.normal([500], dtype=tf.float64)],0)
    def condition(t,y,idx):
      return tf.less(t.read(idx), t_end)
    def body(t,y,idx):
      y_temp = y.read(idx)
      t_temp = t.read(idx)
      if idx == 800:
        y_temp = y_temp + pulse
      if idx == 1600:
        y_temp = y_temp + pulse
      if idx == 3200:
        y_temp = y_temp + pulse
      k1 = h * f(t_temp, y_temp)
      k2 = h * f(t_temp + h / 5, y_temp + k1 / 5)
      k3 = h * f(t_temp + 3 * h / 10, y_temp + 3 * k1 / 40 + 9 * k2 / 40)
      k4 = h * f(t_temp + 4 * h / 5, y_temp + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9)
      k5 = h * f(t_temp + 8 * h / 9, y_temp + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729)
      k6 = h * f(t_temp + h, y_temp + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656)

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
  def adaptive_runge_kutta_dormand_prince(self, f, t0, y0, t_end, h, atol, rtol):
    y = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = tf.TensorArray(dtype=tf.float64, size = 0, dynamic_size=True, clear_after_read=False)
    t = t.write(0,t0)
    y = y.write(0,y0)
    idx = tf.constant(0, dtype=tf.int32)
    def step_integ(f,t,y,h):
      k1 = h * f(t, y)
      k2 = h * f(t + h / 5, y + k1 / 5)
      k3 = h * f(t + 3 * h / 10, y + 3 * k1 / 40 + 9 * k2 / 40)
      k4 = h * f(t + 4 * h / 5, y + 44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9)
      k5 = h * f(t + 8 * h / 9, y + 19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729)
      k6 = h * f(t + h, y + 9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656)

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

  # @tf.function
  def fit(self, fn, y0, t0, t_max, step, fn_params = None, integrator_params = None):
    self.set_initial_cond(y0,t0)
    self.set_pace(t_max, step)
    self.set_dynamics(fn, fn_params) 
    self.integrator_params = integrator_params
  
  # @tf.function
  def run(self, **kwargs):
    if self.fn == None:
      print("Function describing dynamics not found - Run `fit` before this function")
      return 1
    return self.integrator(self.fn, self.t0, self.y0, self.t_max, self.step, **kwargs)

