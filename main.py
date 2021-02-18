#!/usr/bin/env python

"""
File: main.py
Author: Ryan Saltus and Iman Salehi
Email: ryan.saltus@uconn.edu and iman.salehi@uconn.edu
Description: The main code for the OMPC Project.
"""
import numpy as np
from numpy import matmul
from math import exp, log, atanh, tanh
import matplotlib.pyplot as plt

import scipy.io as sio

import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
from baxter_core_msgs.msg import EndpointState

class barrier(object):

    """A class for implementing barrier transformations based on VdP Oscillator dynamics"""

    def __init__(self, a1, a2, A1, A2):
        self.a = np.array([[a1], [a2]])
        self.A = np.array([[A1], [A2]]) 

    def log_transform(self, x):
        """Logarithmic barrier function transformation."""
        
        s = np.log(np.multiply(self.A, (self.a-x))/np.multiply(self.a, self.A-x))
        return s

    def inverse(self, s):
        """Inverts the state in constrained state back to the original space."""

        pre = np.multiply(self.a, self.A)
        num = np.exp(s/2)-np.exp(-s/2)
        den = np.multiply(self.a, np.exp(s/2)) - np.multiply(self.A, np.exp(-s/2))
        
        x = np.multiply(pre, num)
        x = np.divide(x, den)
        
        return x

    def dot_inverse(self, s):
        """Calculates the time derivative of the inverse of the barrier function"""
        num = np.multiply(self.A, self.a**2) - np.multiply(self.a, self.A**2)
        den = np.multiply(self.a**2, np.exp(s)) - 2*np.multiply(self.a, self.A) \
                                           + np.multiply((self.A**2), np.exp(-s))
        return np.divide(num, den) 

    def dyn_transform(self, s):
        """Applies a logarithmic barrier function transformation to unconstrained state
        dynamics, to transform the dynamics into a constrained space."""
        
        num = (self.A[-1,0]**2)*exp(-s[-1,0]) - 2*self.a[-1,0]*self.A[-1,0] \
                                          + (self.a[-1,0]**2)*exp(s[-1,0])
        den = self.A[-1,0]*(self.a[-1,0]**2) - self.a[-1,0]*(self.A[-1,0]**2)
        multiplier = num/den
        
        x = self.inverse(s)
        dotinv = self.dot_inverse(s)[0,0]
        fx = -x[0,0] - 0.5*x[-1,0]*(1-(x[0,0]**2))
        
        # Transformation of state dynamics 
        Gs = np.array([[0], [multiplier*x[0,0]]])
        Fs = np.array([[x[-1,0]/dotinv], [multiplier*fx]])

        return Fs, Gs

class ActorCritic(object):

    """A class implementing the Actor Critic network. """

    def __init__(self, itera):
        
        self.lmbda = 8.0
        self.alpha_a = 1.5
        self.alpha_c = .1 
        self.kappa = 1.0
        self.c = 1.0
        self.dt = .001

        self.R = 1.0
        self.Q = np.identity(2)

        self.num_kernels = 3
        self.Wa = np.random.rand(self.num_kernels,1)
        self.Wc = np.random.rand(self.num_kernels,1)
        
        self.iterations = itera 
        self.s_array = np.zeros((2, self.iterations))
        self.x_array = np.zeros((2, self.iterations))
        self.wa_array = np.zeros((self.num_kernels, self.iterations))
        self.wc_array = np.zeros((self.num_kernels, self.iterations))
        self.Ua_array = np.zeros((1, self.iterations))
        self.xdot = np.zeros((2,1))

    def record_data(self, iteration, s, x, Ua):
        self.s_array[:,iteration] = s[:,0].copy()
        self.x_array[:,iteration] = x[:,0].copy()
        self.wa_array[:,iteration] = self.Wa[:,0].copy()
        self.wc_array[:,iteration] = self.Wc[:,0].copy()
        self.Ua_array[:,iteration] = Ua 

    def offline_update(self, iteration, s, barrier, dt):
       
        #self.dt = dt
        old_x = barrier.inverse(s)
        Fs, Gs = barrier.dyn_transform(s)
        dphi = np.array([[2.0*s[0,0], s[1,0], 0],[ 0, s[0,0], 2.0*s[1,0]]]).T

        Da = (1/(2*self.lmbda))*(1/self.R)*matmul(matmul(Gs.T, dphi.T),  self.Wa)
        ua = -self.lmbda*np.tanh(Da)

        sdot = Fs + Gs*ua
        Dc = (1/(2.0*self.lmbda))*(1/self.R)*matmul(matmul(Gs.T, dphi.T),  self.Wc)
        uc = -self.lmbda*np.tanh(Dc)

        eu = uc - ua
        sigma_a = matmul(dphi, sdot)

        a = 2.0*self.lmbda*self.R*(0.5*self.lmbda*log(self.lmbda**2-ua**2)+ua+atanh(ua/self.lmbda))
        b = 2.0*self.lmbda*self.R*(0.5*self.lmbda*log(self.lmbda**2)+atanh(0/self.lmbda))
        Ua = matmul(matmul(s.T, self.Q), s) + (a-b)
        M = matmul(dphi, Gs)*self.lmbda*(np.tanh(self.kappa*Da)-np.tanh(Da))
        Y = matmul(M, M.T)/2.0 + self.c*np.ones((self.num_kernels, self.num_kernels))

        Wa_dot = -self.alpha_a*(matmul(dphi, Gs)*eu + matmul(dphi, Gs)*(tanh(Da)**2)*eu + matmul(Y, self.Wa))
        self.Wa = self.Wa + self.dt*Wa_dot

        Wc_dot = -self.alpha_c*(np.divide(sigma_a,((1+matmul(sigma_a.T, sigma_a))**2))*(Ua+matmul(self.Wc.T,sigma_a)))
        self.Wc = self.Wc + self.dt*Wc_dot 

        s = s + sdot * self.dt
        x = barrier.inverse(s)
        self.xdot = np.multiply(sdot, barrier.dot_inverse(s))

        self.record_data(iteration, s, x, Ua)

        return s, sdot, self.xdot

class RobotController(object):

    def __init__(self, limb):
        self._control_arm = baxter_interface.limb.Limb(limb)
        self._kin = baxter_kinematics(limb)
    
    def command_velocity(self, command):
        rate = rospy.Rate(100)

        control_joint_names = self._control_arm.joint_names()
        jacob_i = self._kin.jacobian_pseudo_inverse()
        self._vel_command = np.transpose([command[0,0], command[1,0], 0, 0, 0, 0])
        vel = matmul(jacob_i, self._vel_command)
        self._joint_command = {'left_s0':vel[0,0], 'left_s1':vel[0,1], 'left_e0':vel[0,2],
                               'left_e1':vel[0,3], 'left_w0':vel[0,4], 'left_w1':vel[0,5],
                               'left_w2':vel[0,6]}
        self._control_arm.set_joint_velocities(self._joint_command)
        rate.sleep()

def test_barrier():
    nn = ActorCritic()
    btest = barrier(-0.6, -0.2, 0.2, 0.5)
    x = np.array([[-0.55], [-0.1]])
    state = btest.log_transform(x)
    inv_state = btest.inverse(state)
    dyn = btest.dyn_transform(state)
    newstate = nn.online_update(state, btest)

def test_ac():
    
    nn = ActorCritic(100000)
    #bar = barrier(-0.6, -0.2, 0.2, 0.5)
    
    #x = np.array([[-0.3], [0.4]])
    bar = barrier(-0.4, -0.4, 0.1, 0.2)
    
    x = np.array([[-0.3], [-0.1]])
    state = bar.log_transform(x)
    dt = .001 
    count = 0 

    while count < nn.iterations:
        state, state_dot, _ = nn.offline_update(count, state, bar, dt)
        count += 1
    
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    ax1.plot(nn.x_array[0,:], nn.x_array[1,:])
   
    l_x, l_y = [bar.a[0], bar.a[0]], [bar.a[1], bar.A[1]]
    r_x, r_y = [bar.A[0], bar.A[0]], [bar.a[1], bar.A[1]]
    t_x, t_y = [bar.a[0], bar.A[0]], [bar.A[1], bar.A[1]]
    b_x, b_y = [bar.a[0], bar.A[0]], [bar.a[1], bar.a[1]]

    ax1.plot(l_x, l_y, r_x, r_y, t_x, t_y, b_x, b_y, marker='o', color='k')
    ax1.set_xlim([-0.8, 0.3])
    ax1.set_ylim([-0.3, 0.6])

    plt.show()


def test_system():
    rospy.init_node('RL_barrier', anonymous=True) 
    nn = ActorCritic(10000)
    #bar = barrier(-0.4, -0.2, 0.3, 0.2)
    #bar = barrier(-0.2, -0.6, 0.1, 0.6)
    bar = barrier(-0.4, -0.4, 0.1, 0.2)
    
    x = np.array([[-0.3], [-0.1]])
    #x = np.array([[-0.3], [0.1]])

    rc = RobotController('left')  

    initial = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
    init_x = initial.pose.position.x 
    init_y = initial.pose.position.y 
    then = rospy.get_rostime()
    then = then.to_sec()
    then = float(then)
    prev_time = initial.header.stamp.secs + initial.header.stamp.nsecs*(10**-9)

    transform_vector = np.array([[init_x - x[0,0]],[init_y-x[1,0]]])
    state = bar.log_transform(x)
    count = 0 

    while count < nn.iterations:

        curr_state = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
        now = rospy.get_rostime()
        now = now.to_sec()
        now = float(now)
        feedback_x = curr_state.pose.position.x
        feedback_y = curr_state.pose.position.y
        dt = now - then

        feedback_state = np.array([[feedback_x],[feedback_y]])
        feedback_state = feedback_state - transform_vector
        feedback_state = bar.log_transform(feedback_state)

        _, _, xdot = nn.offline_update(count, feedback_state, bar, dt)
        
        rc.command_velocity(xdot)
        count += 1
        then = now
    
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    ax1.plot(nn.x_array[0,:], nn.x_array[1,:])
    l_x, l_y = [bar.a[0], bar.a[0]], [bar.a[1], bar.A[1]]
    r_x, r_y = [bar.A[0], bar.A[0]], [bar.a[1], bar.A[1]]
    t_x, t_y = [bar.a[0], bar.A[0]], [bar.A[1], bar.A[1]]
    b_x, b_y = [bar.a[0], bar.A[0]], [bar.a[1], bar.a[1]]

    ax1.plot(l_x, l_y, r_x, r_y, t_x, t_y, b_x, b_y, marker='o', color='k')
    ax1.set_xlim([-0.8, 0.3])
    ax1.set_ylim([-0.3, 0.6])
    
    fig2 = plt.figure() 
    
    ax2 = fig2.add_subplot(111)
    ax2.plot(nn.wa_array[0,:])
    ax2.plot(nn.wa_array[1,:])
    ax2.plot(nn.wa_array[2,:])
    
    ax3 = fig2.add_subplot(111)
    ax3.plot(nn.wc_array[0,:])
    ax3.plot(nn.wc_array[1,:])
    ax3.plot(nn.wc_array[2,:])

    fig3 = plt.figure()
    ax1.plot(nn.Ua_array[0,:])

    plt.show()
    
    sio.savemat('critic_weights.mat', {'wc':nn.wc_array})
    sio.savemat('actor_weights.mat', {'wa':nn.wa_array})
    sio.savemat('demo_w_barrier.mat', {'x_bar':nn.x_array})
    sio.savemat('reward.mat', {'Ua':nn.Ua_array})

def test_dynamics():
    rospy.init_node('RL_barrier', anonymous=True) 
    #bar = barrier(-0.4, -0.2, 0.3, 0.2)
    #bar = barrier(-0.2, -0.6, 0.1, 0.6)
    bar = barrier(-0.4, -0.4, 0.1, 0.3)
    
    x = np.array([[-0.3], [-0.1]])
    #x = np.array([[-0.3], [0.1]])

    rc = RobotController('left')  

    initial = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
    init_x = initial.pose.position.x 
    init_y = initial.pose.position.y 
    then = rospy.get_rostime()
    then = then.to_sec()
    then = float(then)
    prev_time = initial.header.stamp.secs + initial.header.stamp.nsecs*(10**-9)

    transform_vector = np.array([[init_x - x[0,0]],[init_y-x[1,0]]])
    state = bar.log_transform(x)
    count = 0 
    iterations = 10000
    record_array = np.zeros((2,iterations))

    while count < iterations:

        curr_state = rospy.wait_for_message('/robot/limb/left/endpoint_state', EndpointState)
        now = rospy.get_rostime()
        now = now.to_sec()
        now = float(now)
        feedback_x = curr_state.pose.position.x
        feedback_y = curr_state.pose.position.y
        dt = now - then

        feedback_state = np.array([[feedback_x],[feedback_y]])
        feedback_state = feedback_state - transform_vector
        
        record_array[:, count] = feedback_state[:,0]
        
        sy, sx = feedback_state[1,0], feedback_state[0,0]

        ustar = -sy*sx
        x1_d = sy 
        x2_d = -sx-.5*(1-sx**2)*sy + sx*ustar;  
        
        xdot = np.array([[x1_d],[x2_d]]) 
        xdot = xdot/10.0
        
        rc.command_velocity(xdot)
        count += 1
        then = now
    
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    ax1.plot(record_array[0,:], record_array[1,:])
    l_x, l_y = [bar.a[0], bar.a[0]], [bar.a[1], bar.A[1]]
    r_x, r_y = [bar.A[0], bar.A[0]], [bar.a[1], bar.A[1]]
    t_x, t_y = [bar.a[0], bar.A[0]], [bar.A[1], bar.A[1]]
    b_x, b_y = [bar.a[0], bar.A[0]], [bar.a[1], bar.a[1]]

    ax1.plot(l_x, l_y, r_x, r_y, t_x, t_y, b_x, b_y, marker='o', color='k')
    ax1.set_xlim([-0.8, 0.3])
    ax1.set_ylim([-0.3, 0.6])
    
    plt.show()
    sio.savemat('demo_wo_barrier.mat', {'x_nbar':record_array})

if __name__ == "__main__":
    test_system()
    #test_dynamics()
    #test_ac()
