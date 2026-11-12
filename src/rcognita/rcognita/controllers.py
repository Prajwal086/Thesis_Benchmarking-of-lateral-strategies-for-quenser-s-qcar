#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from .utilities import dss_sim
from .utilities import rep_mat
from .utilities import uptria2vec
from .utilities import push_vec
from . import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from numpy.linalg import lstsq
from numpy import reshape
import warnings

# For debugging purposes
from tabulate import tabulate

try:
    import sippy
except ModuleNotFoundError:
    warnings.warn_explicit('\nImporting sippy failed. You may still use rcognita, but' +
                  ' without model identification capability. \nRead on how' +
                  ' to install sippy at https://github.com/AIDynamicAction/rcognita\n', 
                  UserWarning, __file__, 33)


def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, mode):
    """
    Main interface for various controllers.

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method.

    Returns
    -------
    action : : array of shape ``[dim_input, ]``.
        Control action.

    """
    
    if mode=='manual': 
        action = action_manual
    elif mode=='nominal': 
        action = ctrl_nominal.compute_action(t, observation)
    else: # Controller for benchmakring
        action = ctrl_benchmarking.compute_action(t, observation)
        
    return action

class CtrlRLStab:
    """
    Class of reinforcement learning agents with stabilizing constraints.
    
    Sampling here is similar to the predictive controller agent ``CtrlOptPred``
    
    Needs a nominal controller object ``safe_ctrl`` with a respective Lyapunov function.
    
    Actor
    -----
    
    ``w_actor`` : weights.

    Feature structure is defined via a string flag ``actor_struct``. Read more on features in class description of ``controllers.CtrlOptPred``.
    
    Critic
    -----
    
    ``w_critic`` : weights.
    
    Feature structure is defined via a string flag ``critic_struct``. Read more on features in class description of ``controllers.CtrlOptPred``.
    
    Attributes
    ----------
    mode : : string
        Controller mode. Currently available only JACS, joint actor-critic (stabilizing).   
    
    Read more
    ---------

    Osinenko, P., Beckenbach, L., Göhrt, T., & Streif, S. (2020). A reinforcement learning method with closed-loop stability guarantee. IFAC-PapersOnLine  
    
    """

    def __init__(self,
                 dim_input,
                 dim_output,
                 mode = 'JACS',
                 ctrl_bnds = [],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 is_est_model=0,
                 model_est_stage=1,
                 model_est_period=0.1,
                 buffer_size=20,
                 model_order=3,
                 model_est_checks=0,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 actor_struct='quad-nomix',
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[],   
                 safe_ctrl=[],
                 safe_decay_rate=[]):
        
        """
        Parameter specification largely resembles that of ``CtrlOptPred`` class.
        
        Parameters
        ----------
        dim_input, dim_output : : integer
            Dimension of input and output which should comply with the system-to-be-controlled.  
    
        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default).
        action_init : : array of shape ``[dim_input, ]``   
            Initial action to initialize optimizers.         
        t0 : : number
            Initial value of the controller's internal clock.
        sampling_time : : number
            Controller's sampling time (in seconds).
        sys_rhs, sys_out : : functions        
            Functions that represent the right-hand side, resp., the output of the exogenously passed model.
            The latter could be, for instance, the true model of the system.
            In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
            Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
        prob_noise_pow : : number
            Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
        is_est_model : : number
            Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
        model_est_stage : : number
            Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
        model_est_period : : number
            Time between model estimate updates (in seconds).
        buffer_size : : natural number
            Size of the buffer to store data.
        model_order : : natural number
            Order of the state-space estimation model
            
            .. math::
                \\begin{array}{ll}
        			\\hat x^+ & = A \\hat x + B action, \\newline
        			observation^+  & = C \\hat x + D action.
                \\end{array}             
            
            See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
            When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
            neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``. 
        model_est_checks : : natural number
            Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
            May improve the prediction quality somewhat.
        gamma : : number in (0, 1]
            Discounting factor.
            Characterizes fading of stage objectives along horizon.
        Ncritic : : natural number
            Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
            optimal infinite-horizon objective (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
        critic_period : : number
            The same meaning as ``model_est_period`.` 
        critic_struct, actor_struct : : string
            Choice of the structure of the critic's and actor's features.
            
            Currently available:
                
            .. list-table:: Feature structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quad-lin'
                 - Quadratic-linear
               * - 'quadratic'
                 - Quadratic
               * - 'quad-nomix'
                 - Quadratic, no mixed terms
           
            *Add your specification into the table when customizing the actor and critic*. 
        stage_obj_struct : : string
            Choice of the stage objective structure.
            
            Currently available:
               
            .. list-table:: Running objective structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quadratic'
                 - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
               * - 'biquadratic'
                 - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
                   should be ``[R1, R2]``
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        
        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)            
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys
        
        # Model estimator's things
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.model_est_stage = model_est_stage
        self.model_est_period = model_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.model_est_checks = model_est_checks
        
        A = np.zeros( [self.model_order, self.model_order] )
        B = np.zeros( [self.model_order, self.dim_input] )
        C = np.zeros( [self.dim_output, self.model_order] )
        D = np.zeros( [self.dim_output, self.dim_input] )
        x0est = np.zeros( self.model_order )
        
        self.my_model = models.ModelSS(A, B, C, D, x0est)
        
        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)        
        
        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.actor_struct = actor_struct
        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( (  self.dim_output  + 1 ) *  self.dim_output / 2 + self.dim_output )
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( self.dim_output + 1 ) * self.dim_output / 2 ).astype(int)
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)
 
        self.w_critic_prev = self.Wmin
        self.w_critic_init = np.ones(self.dim_critic)
        
        self.lmbd_prev = 0
        self.lmbd_init = 0
        
        self.lmbd_min = 0
        self.lmbd_max = 1
           
        if self.actor_struct == 'quad-lin':
            self.dim_actor_per_input = int( ( self.dim_output  + 1 ) *  self.dim_output / 2 + self.dim_output ) 
        elif self.actor_struct == 'quadratic':
            self.dim_actor_per_input = int( ( self.dim_output + 1 ) * self.dim_output / 2 )
        elif self.actor_struct == 'quad-nomix':
            self.dim_actor_per_input = self.dim_output
          
        self.dim_actor = self.dim_actor_per_input * self.dim_input   
            
        self.Hmin = -0.5e1*np.ones(self.dim_actor) 
        self.Hmax = 0.5e1*np.ones(self.dim_actor)   
 
        # Stabilizing constraint stuff
        self.safe_ctrl = safe_ctrl          # Safe controller (agent)
        self.safe_decay_rate = safe_decay_rate

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        stage_obj = 0

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return stage_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time
    
    def _actor(self, observation, w_actor):
        """
        Actor: a routine that models the policy.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.actor_struct == 'quad-lin':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ), observation ])
        elif self.actor_struct == 'quadratic':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ) ])   
        elif self.actor_struct == 'quad-nomix':
            regressor_actor = observation * observation        

        return reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ regressor_actor

    def _critic(self, observation, w_critic, lmbd):
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        The parameter ``lmbd`` is needed here specifically for joint actor-critic (stabilizing) a.k.a. JACS.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = observation
        else:
            chi = observation - self.observation_target
        
        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi

        return lmbd * w_critic @ regressor_critic + ( 1 - lmbd ) * self.safe_ctrl.compute_LF( observation )

    def _w_actor_from_action(self, action, observation):
        """
        Compute actor weights from a given action.
        
        The current implementation is for linearly parametrized models so far.

        """
        
        if self.actor_struct == 'quad-lin':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ), observation ])
        elif self.actor_struct == 'quadratic':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ) ])   
        elif self.actor_struct == 'quad-nomix':
            regressor_actor = observation * observation          
        
        return reshape(lstsq( np.array( [ regressor_actor ] ), np.array( [ action ] ) )[0].T, self.dim_actor ) 

    def _actor_critic_cost(self, w_all):
        """
        Cost (loss) of joint actor-critic (stabilizing) a.k.a. JACS
       
        """        
        
        observation_sqn = self.observation_buffer[-self.Ncritic:,:]
        
        w_critic = w_all[:self.dim_critic]
        # lmbd = w_all[self.dim_critic+1]
        w_actor = w_all[-self.dim_actor:]         
        
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            observation_prev = observation_sqn[k-1, :]
            observation_next = observation_sqn[k, :]
            
            critic_prev = self._critic(observation_prev, w_critic, 1)
            critic_next = self._critic(observation_next, self.w_critic_prev, 1)
            
            action = self._actor(observation_prev, w_actor)
            
            # Temporal difference
            e = critic_prev - self.gamma * critic_next - self.stage_obj(observation_prev, action)
            
            Jc += 1/2 * e**2
        
        return Jc

    def _actor_critic_optimizer(self, observation):
        """
        This method is effectively a wrapper for an optimizer that minimizes :func:`~controllers.CtrlRLStab._actor_critic_cost`.
        It implements the stabilizing constraints.
        
        The variable ``w_all`` here is a stack of actor, critic and auxiliary critic weights.
        
        Important remark: although the current implementation concentrates on a joint coss (loss) of actor-critic (see :func:`~controllers.CtrlRLStab._actor_critic_cost`),
        nothing stops us from doing a usual split actor-critic training.
        The key point of CtrlRLStab agent is its stabilizing constraints that can actually be invoked as a filter (a safety checker), that replaces the action and 
        critic parameters for safe ones if any of the stabilizing constraints are violated.

        """  

        def constr_stab_par_decay(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_curr = self._critic(observation, self.w_critic_prev, self.lmbd_prev)   
            critic_new = self._critic(observation, w_critic, lmbd)
            
            return critic_new - critic_curr
            
        def constr_stab_LF_bound(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:] 
                        
            action = self._actor(observation, w_actor)
            
            observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)  # Euler scheme
            
            critic_next = self._critic(observation_next, w_critic, lmbd) 
            
            return self.safe_ctrl.compute_LF(observation_next) - critic_next        
        
        def constr_stab_decay(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:]   
            
            action = self._actor(observation, w_actor)
            
            observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)  # Euler scheme
            
            critic_new = self._critic(observation, w_critic, lmbd)   
            critic_next = self._critic(observation_next, w_critic, lmbd)   
            
            return critic_next - critic_new + self.safe_decay_rate

        def constr_stab_positive(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_new = self._critic(observation, w_critic, lmbd)
            
            return - critic_new

        # Constraint violation tolerance
        eps1 = 1e-3
        eps2 = 1e-3
        eps3 = 1e-3
        eps4 = 1e-3
        
        # my_constraints = (
        #     NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, observation ), -np.inf, eps1, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, observation ), -np.inf, eps2, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, observation ), -np.inf, eps3, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, observation ), -np.inf, eps4, keep_feasible=True)
        #     )
        
        my_constraints = (
            NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, observation ), -np.inf, eps1),
            NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, observation ), -np.inf, eps2),
            NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, observation ), -np.inf, eps3),
            NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, observation ), -np.inf, eps4)
            )        

        # Optimization methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        opt_method = 'SLSQP'
        if opt_method == 'trust-constr':
            opt_options = {'maxiter': 10, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            opt_options = {'maxiter': 10, 'maxfev': 10, 'disp': False, 'adaptive': True, 'xatol': 1e-4, 'fatol': 1e-4} # 'disp': True, 'verbose': 2} 
        
        # Bounds are not practically necessary for stabilizing joint actor-critic to function
        # bnds = sp.optimize.Bounds(np.hstack([self.Wmin, self.lmbd_min, self.Hmin]), 
        #                           np.hstack([self.Wmax, self.lmbd_max, self.Hmax]), 
        #                           keep_feasible=True)
        
        # ToDo: make a better routine to determine initial actor weights for the given action
        self.w_actor_init = self._w_actor_from_action( self.safe_ctrl.compute_action_vanila( observation ), observation )
        
        # DEBUG ===================================================================
        # ================================Constraint debugger

        # w_all = np.concatenate([self.w_critic_init, np.array([self.lmbd_init]), self.w_actor_init])
        
        # w_critic = w_all[:self.dim_critic]
        # lmbd = w_all[self.dim_critic]
        # w_actor = w_all[-self.dim_actor:] 
                    
        # action = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._regressor_actor( observation )
        
        # constr_stab_par_decay(w_all, observation)
        # constr_stab_LF_bound(w_all, observation)
        # constr_stab_decay(w_all, observation)
        # constr_stab_positive(w_all, observation)

        # /DEBUG ===================================================================         
        
        # Notice `bounds=bnds` is removed from arguments of minimize.
        # It is because bounds are not practically necessary for stabilizing joint actor-critic to function
        # w_all = minimize(self._actor_critic_cost,
        #                     np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.w_actor_init]),
        #                     method=opt_method, tol=1e-4, constraints=my_constraints, options=opt_options).x
        
        w_all = minimize(self._actor_critic_cost,
                            np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.w_actor_init]),
                            method=opt_method,
                            tol=1e-4,
                            options=opt_options).x        
        
        w_critic = w_all[:self.dim_critic]
        lmbd = w_all[self.dim_critic]
        w_actor = w_all[-self.dim_actor:]       
        
        action = self._actor(observation, w_actor)
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        # dataRow = [constr_stab_par_decay(w_all, observation), constr_stab_LF_bound(w_all, observation), constr_stab_decay(w_all, observation), constr_stab_positive(w_all, observation)]
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ===================================================================        
        
        # Safety checker!
        if constr_stab_par_decay(w_all, observation) >= eps1 or \
            constr_stab_LF_bound(w_all, observation) >= eps2 or \
            constr_stab_decay(w_all, observation) >= eps3 or \
            constr_stab_positive(w_all, observation) >= eps4 :
                
            w_critic = self.w_critic_init
            lmbd = self.lmbd_init

            action = self.safe_ctrl.compute_action_vanila( observation )
            
            w_actor = self._w_actor_from_action( action, observation )
       
        # DEBUG ===================================================================   
        # ================================Put safe controller through        
        # w_critic = self.w_critic_init
        # lmbd = self.lmbd_init
        # action = self.safe_ctrl.compute_action_vanila(observation)        
        # /DEBUG ===================================================================         
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        # dataRow = [constr_stab_par_decay(w_all, observation), constr_stab_LF_bound(w_all, observation), constr_stab_decay(w_all, observation), constr_stab_positive(w_all, observation)]
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ===================================================================  
        
        # STUB ===================================================================   
        # ================================Optimization of one stage_obj + LF_next
        # def J_tmp(action, observation):
        #     observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)
        #     return self.safe_ctrl.compute_LF(observation_next) + self.stage_obj(observation_next, action) 
        #     # return self.safe_ctrl.compute_LF(observation_next)
        
        # action = minimize(lambda action: J_tmp(action, observation),
        #               np.zeros(2),
        #               method=opt_method, tol=1e-6, options=opt_options).x        
        
        # /STUB ===================================================================
        
        return w_critic, lmbd, action
        
    def compute_action(self, t, observation):

        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            # Update data buffers
            self.action_buffer = push_vec(self.action_buffer, self.action_curr)
            self.observation_buffer = push_vec(self.observation_buffer, observation)          
            
            w_critic, lmbd, action = self._actor_critic_optimizer(observation)
            
            self.w_critic_prev = w_critic            
            self.lmbd_prev = lmbd

            for k in range(2):
                action[k] = np.clip(action[k], self.action_min[k], self.action_max[k]) 

            self.action_curr = action

            return action
        
        else:
            return self.action_curr        

class CtrlOptPred:
    """
    Class of predictive optimal controllers, primarily model-predictive control and predictive reinforcement learning, that optimize a finite-horizon cost.
    
    Currently, the actor model is trivial: an action is generated directly without additional policy parameters.
        
    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled.
    mode : : string
        Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):
          
        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1
    
           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
             - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
           * - 'SQL' - RL/ADP via stacked Q-learning
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat \\gamma^{k-1} Q^{\\theta}(y_{N_a}, u_{N_a})`               
        
        Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
        
        *Add your specification into the table when customizing the agent*.    

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default).
    action_init : : array of shape ``[dim_input, ]``   
        Initial action to initialize optimizers.          
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`. 
    pred_step_size : : number
        Prediction step size in :math:`J_a` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon.
    sys_rhs, sys_out : : functions        
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
    is_est_model : : number
        Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
    model_est_stage : : number
        Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
    model_est_period : : number
        Time between model estimate updates (in seconds).
    buffer_size : : natural number
        Size of the buffer to store data.
    model_order : : natural number
        Order of the state-space estimation model
        
        .. math::
            \\begin{array}{ll}
    			\\hat x^+ & = A \\hat x + B action, \\newline
    			observation^+  & = C \\hat x + D action.
            \\end{array}             
        
        See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
        When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``. 
    model_est_checks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
        May improve the prediction quality somewhat.
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of stage objectives along horizon.
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
    critic_period : : number
        The same meaning as ``model_est_period``. 
    critic_struct : : natural number
        Choice of the structure of the critic's features.
        
        Currently available:
            
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector
       
        *Add your specification into the table when customizing the critic*. 
    stage_obj_struct : : string
        Choice of the stage objective structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
               should be ``[R1, R2]``   
        
        *Pass correct stage objective parameters in* ``stage_obj_pars`` *(as a list)*
        
        *When customizing the stage objective, add your specification into the table above*
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """    
         
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 is_est_model=0,
                 model_est_stage=1,
                 model_est_period=0.1,
                 buffer_size=20,
                 model_order=3,
                 model_est_checks=0,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[]):
        """
        Parameters
        ----------
        dim_input, dim_output : : integer
            Dimension of input and output which should comply with the system-to-be-controlled.
        mode : : string
            Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):
              
            .. list-table:: Controller modes
               :widths: 75 25
               :header-rows: 1
        
               * - Mode
                 - Cost function
               * - 'MPC' - Model-predictive control (MPC)
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
               * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
                 - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
               * - 'SQL' - RL/ADP via stacked Q-learning
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`               
            
            Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
            
            *Add your specification into the table when customizing the agent* .   
    
        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default).
        action_init : : array of shape ``[dim_input, ]``   
            Initial action to initialize optimizers.              
        t0 : : number
            Initial value of the controller's internal clock
        sampling_time : : number
            Controller's sampling time (in seconds)
        Nactor : : natural number
            Size of prediction horizon :math:`N_a` 
        pred_step_size : : number
            Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
            convenience. Larger prediction step size leads to longer factual horizon.
        sys_rhs, sys_out : : functions        
            Functions that represent the right-hand side, resp., the output of the exogenously passed model.
            The latter could be, for instance, the true model of the system.
            In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
            Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
        prob_noise_pow : : number
            Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
        is_est_model : : number
            Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
        model_est_stage : : number
            Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
        model_est_period : : number
            Time between model estimate updates (in seconds).
        buffer_size : : natural number
            Size of the buffer to store data.
        model_order : : natural number
            Order of the state-space estimation model
            
            .. math::
                \\begin{array}{ll}
        			\\hat x^+ & = A \\hat x + B action, \\newline
        			observation^+  & = C \\hat x + D action.
                \\end{array}             
            
            See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
            When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
            neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons`` 
        model_est_checks : : natural number
            Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
            May improve the prediction quality somewhat.
        gamma : : number in (0, 1]
            Discounting factor.
            Characterizes fading of stage objectives along horizon.
        Ncritic : : natural number
            Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
            optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
        critic_period : : number
            The same meaning as ``model_est_period``. 
        critic_struct : : natural number
            Choice of the structure of the critic's features.
            
            Currently available:
                
            .. list-table:: Critic feature structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quad-lin'
                 - Quadratic-linear
               * - 'quadratic'
                 - Quadratic
               * - 'quad-nomix'
                 - Quadratic, no mixed terms
               * - 'quad-mix'
                 - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
                   where :math:`w` is the critic's weights
           
            *Add your specification into the table when customizing the critic*.
        stage_obj_struct : : string
            Choice of the stage objective structure.
            
            Currently available:
               
            .. list-table:: Running objective structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quadratic'
                 - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
               * - 'biquadratic'
                 - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
                   should be ``[R1, R2]``
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        
        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys
        
        # Model estimator's things
        self.is_est_model = is_est_model
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.model_est_stage = model_est_stage
        self.model_est_period = model_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.model_est_checks = model_est_checks
        
        A = np.zeros( [self.model_order, self.model_order] )
        B = np.zeros( [self.model_order, self.dim_input] )
        C = np.zeros( [self.dim_output, self.model_order] )
        D = np.zeros( [self.dim_output, self.dim_input] )
        x0est = np.zeros( self.model_order )
        
        self.my_model = models.ModelSS(A, B, C, D, x0est)
        
        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)        
        
        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) ) 
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)
            
        self.w_critic_prev = np.ones(self.dim_critic)  
        self.w_critic_init = self.w_critic_prev
        
        # self.big_number = 1e4

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        stage_obj = 0

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return stage_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time
    
    def _estimate_model(self, t, observation):
        """
        Estimate model parameters by accumulating data buffers ``action_buffer`` and ``observation_buffer``.
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update buffers when using RL or requiring estimated model
            if self.is_est_model or self.mode in ['RQL', 'SQL']:
                time_in_est_period = t - self.est_clock
                
                # Estimate model if required
                if (time_in_est_period >= self.model_est_period) and self.is_est_model:
                    # Update model estimator's internal clock
                    self.est_clock = t
                    
                    try:
                        # Using Github:CPCLAB-UNIPI/SIPPY 
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(self.observation_buffer, self.action_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.buffer_size/12),
                                                            # SS_p=int(self.buffer_size/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sampling_time)
                        
                        self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # ToDo: train an NN via Torch
                        # NN_wgts = NN_train(...)
                        
                    except:
                        print('Model estimation problem')
                        self.my_model.upd_pars(np.zeros( [self.model_order, self.model_order] ),
                                                np.zeros( [self.model_order, self.dim_input] ),
                                                np.zeros( [self.dim_output, self.model_order] ),
                                                np.zeros( [self.dim_output, self.dim_input] ) )
                    
                    # Model checks
                    if self.model_est_checks > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        tot_abs_err_curr = 1e8
                        for k in range(self.model_est_checks):
                            A, B, C, D = self.model_stack[k].A, self.model_stack[k].B, self.model_stack[k].C, self.model_stack[k].D
                            x0est,_,_,_ = np.linalg.lstsq(C, observation)
                            Yest,_ = dss_sim(A, B, C, D, self.action_buffer, x0est, observation)
                            mean_err = np.mean(Yest - self.observation_buffer, axis=0)
                            
                            # DEBUG ===================================================================
                            # ================================Interm output of model prediction quality
                            # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                            # dataRow = []
                            # for k in range(dim_output):
                            #     dataRow.append( mean_err[k] )
                            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                            # print( table )
                            # /DEBUG ===================================================================
                            
                            tot_abs_err = np.sum( np.abs( mean_err ) )
                            if tot_abs_err <= tot_abs_err_curr:
                                tot_abs_err_curr = tot_abs_err
                                self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # DEBUG ===================================================================
                        # ==========================================Print quality of the best model
                        # R  = '\033[31m'
                        # Bl  = '\033[30m'
                        # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, observation)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.action_buffer, x0est, observation)
                        # mean_err = np.mean(Yest - ctrlStat.observation_buffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                        # dataRow = []
                        # for k in range(dim_output):
                        #     dataRow.append( mean_err[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                        # print(R+table+Bl)
                        # /DEBUG ===================================================================                    
            
            # Update initial state estimate
            x0est,_,_,_ = np.linalg.lstsq(self.my_model.C, observation)
            self.my_model.updateIC(x0est)
     
            if t >= self.model_est_stage:
                    # Drop probing noise
                    self.is_prob_noise = 0 

    def _critic(self, observation, action, w_critic):
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi
        elif self.critic_struct == 'quad-mix':
            regressor_critic = np.concatenate([ observation**2, np.kron(observation, action), action**2 ]) 

        return w_critic @ regressor_critic
    
    def _critic_cost(self, w_critic):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            observation_prev = self.observation_buffer[k-1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k-1, :]
            action_next = self.action_buffer[k, :]
            
            # Temporal difference
            
            critic_prev = self._critic(observation_prev, action_prev, w_critic)
            critic_next = self._critic(observation_next, action_next, self.w_critic_prev)            

            e = critic_prev - self.gamma * critic_next - self.stage_obj(observation_prev, action_prev)
            
            Jc += 1/2 * e**2
            
        return Jc
        
        
    def _critic_optimizer(self):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._critic_cost`.

        """        
        
        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
        
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)
    
        w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x
        
        # DEBUG ===================================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( w_critic )
        # /DEBUG ==================================================================
        
        return w_critic
    
    def _actor_cost(self, action_sqn, observation):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """
        
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System output prediction
        if not self.is_est_model:    # Via exogenously passed model
            observation_sqn[0, :] = observation
            state = self.state_sys
            for k in range(1, self.Nactor):
                # state = get_next_state(state, my_action_sqn[k-1, :], delta)         TODO
                state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])  # Euler scheme
                
                observation_sqn[k, :] = self.sys_out(state)

        elif self.is_est_model:    # Via estimated model
            my_action_sqn_upsampled = my_action_sqn.repeat(int(self.pred_step_size/self.sampling_time), axis=0)
            observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
            observation_sqn = observation_sqn_upsampled[::int(self.pred_step_size/self.sampling_time)]
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
        elif self.mode=='RQL':     # RL: Q-learning with Ncritic-1 roll-outs of stage objectives
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
             J += self._critic(observation_sqn[-1, :], my_action_sqn[-1, :], self.w_critic)
        elif self.mode=='SQL':     # RL: stacked Q-learning
             for k in range(self.Nactor): 
                Q = self._critic(observation_sqn[k, :], my_action_sqn[k, :], self.w_critic)
                
                # With state constraints via indicator function
                # Q = w_critic @ self._regressor_critic( observation_sqn[k, :], my_action_sqn[k, :] ) + state_constraint_indicator(observation_sqn[k, 0])
                
                # DEBUG ===================================================================
                # =========================================================================
                # R  = '\033[31m'
                # Bl  = '\033[30m'
                # if state_constraint_indicator(observation_sqn[k, 0]) > 1:
                #     print(R+str(state_constraint_indicator(observation_sqn[k, 0]))+Bl)
                # /DEBUG ==================================================================                 
                
                J += Q 

        return J
    
    def _actor_optimizer(self, observation):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._actor_cost`.
        See class documentation.
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~controllers.CtrlOptPred._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm.

        """

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(action_sqn, idx):
            
        #     my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])
            
        #     observation_sqn = np.zeros([idx, self.dim_output])    
            
        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         observation_sqn[0, :] = observation
        #         state = self.state_sys
        #         Y[0, :] = observation
        #         x = self.x_s
        #         for k in range(1, idx):
        #             # state = get_next_state(state, my_action_sqn[k-1, :], delta)
        #             state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #             observation_sqn[k, :] = self.sys_out(state)            
            
        #     return observation_sqn[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda action_sqn: state_constraint(action_sqn, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        # actor_opt_method = 'SLSQP' # Standard
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
       
        isGlobOpt = 0
        
        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])
        
        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)
        
        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-7, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-7,
                                      bounds=bnds,
                                      options=actor_opt_options).x        

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = my_action_sqn_init
        
        # DEBUG ===================================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])    
        # my_action_sqn_upsampled = my_action_sqn.repeat(int(delta/self.sampling_time), axis=0)
        # observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
        # observation_sqn = observation_sqn_upsampled[::int(delta/self.sampling_time)]
        # Yt = np.zeros([N, self.dim_output])
        # Yt[0, :] = observation
        # state = self.state_sys
        # for k in range(1, Nactor):
        #     state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sys_out(state)           
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
        # dataRow = []
        # for k in range(dim_output):
        #     dataRow.append( np.mean(observation_sqn[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ==================================================================     
        
        return action_sqn[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, observation):
        """
        Main method. See class documentation.
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~controllers.CtrlOptPred._actor_cost`, here.

        """       
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            if self.mode == 'MPC':  
                
                # Apply control when model estimation phase is over  
                if self.is_prob_noise and self.is_est_model:
                    return self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                
                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation)

                elif self.mode=='MPC':
                    action = self._actor_optimizer(observation)
                    
            elif self.mode in ['RQL', 'SQL']:
                # Critic
                timeInCriticPeriod = t - self.critic_clock
                
                # Update data buffers
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)
                self.observation_buffer = push_vec(self.observation_buffer, observation)
                
                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t
                    
                    self.w_critic = self._critic_optimizer()
                    self.w_critic_prev = self.w_critic
                    
                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.w_critic_init = self.w_critic_prev
                    
                else:
                    self.w_critic = self.w_critic_prev
                    
                # Actor. Apply control when model estimation phase is over
                if self.is_prob_noise and self.is_est_model:
                    action = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation)
                    
                elif self.mode in ['RQL', 'SQL']:
                    action = self._actor_optimizer(observation) 
            
            self.action_curr = action
            
            return action    
    
        else:
            return self.action_curr
        


class CtrlNominal3WRobot:
    """
    This is a class of nominal controllers for 3-wheel robots used for benchmarking of other controllers.
    
    The controller is sampled.
    
    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_]).
  
    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot.
    ctrl_gain : : number
        Controller gain.       
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).       
    
    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594
        
    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887
       
    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)
    
    """
    
    def __init__(self, m, I, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        self.action_curr = np.zeros(2)
   
    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_curr = np.zeros(2)   
    
    def _zeta(self, xNI, theta):
        """
        Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators).

        """
        
        #                                 3
        #                             |x |
        #         4     4             | 3|          
        # L(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta   
        #                     /     / 2   2 \             \ 2
        #                    | sqrt| x + x   | + sqrt|x |  |
        #                     \     \ 1   2 /        | 3| /  
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                         3
        #                                     |x |
        #            4     4                     | 3|          
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #            1     2    
        #                        /                                     \ 2
        #                        | x cos theta + x sin theta + sqrt|x | |
        #                        \ 1             2                | 3| /
        #                           \_______________  ______________/
        #                                            \/
        #                                            sigma~ 
    
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3
        
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3
        
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3  
    
        return nablaF
    
    def _kappa(self, xNI, theta): 
        """
        Stabilizing controller for NI-part.

        """
        kappa_val = np.zeros(2)
        
        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])
                         
        zeta_val = self._zeta(xNI, theta)
        
        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )
        
        return kappa_val
    
    def _Fc(self, xNI, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation.

        """
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        
        z = eta - self._kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)
    
    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0
        
        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
        
        options = {'maxiter': 50, 'disp': False}
        
        theta_val = minimize(lambda theta: self._Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x
        
        return theta_val
        
    def _Cart2NH(self, coords_Cart): 
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """
        
        xNI = np.zeros(3)
        eta = np.zeros(2)
        
        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]
        
        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )
        
        eta[0] = omega
        eta[1] = ( yc * np.cos(alpha) - xc * np.sin(alpha) ) * omega + v   
        
        return [xNI, eta]
  
    def _NH2ctrl_Cart(self, xNI, eta, uNI): 
        """
        Get control for Cartesian NI from NH coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
        

        """

        uCart = np.zeros(2)
        
        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
        uCart[1] = self.I * uNI[0]
        
        return uCart

    def compute_action(self, t, observation):
        """
        See algorithm description in [[1]_], [[2]_].
        
        **This algorithm needs full-state measurement of the robot**.
        
        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887
           
        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t
            
            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH( observation ) 
            theta_star = self._minimizer_theta(xNI, eta)
            kappa_val = self._kappa(xNI, theta_star)
            z = eta - kappa_val
            uNI = - self.ctrl_gain * z
            action = self._NH2ctrl_Cart(xNI, eta, uNI)
            
            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])           
            
            self.action_curr = action

            # DEBUG ===================================================================   
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']  
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
            # print(R+table+Bl)
            # /DEBUG ===================================================================             

            return action    
    
        else:
            return self.action_curr

    def compute_action_vanila(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobot.compute_action`, but without invoking the internal clock.

        """
        
        xNI, eta = self._Cart2NH( observation ) 
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = - self.ctrl_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)
        
        self.action_curr = action
        
        return action

    def compute_LF(self, observation):
        
        xNI, eta = self._Cart2NH( observation ) 
        theta_star = self._minimizer_theta(xNI, eta)
        
        return self._Fc(xNI, eta, theta_star)
    
class CtrlNominal3WRobotNI:
    """
    Nominal parking controller for NI using disassembled subgradients.
    
    """
    
    def __init__(self, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        self.action_curr = np.zeros(2)
   
    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_curr = np.zeros(2)   
    
    def _zeta(self, xNI):
        """
        Analytic disassembled subgradient, without finding minimizer theta.

        """
        
        #                                 3
        #                             |x |
        #         4     4             | 3|          
        # L(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta   
        #                     /     / 2   2 \             \ 2
        #                     | sqrt| x + x   | + sqrt|x | |
        #                     \     \ 1   2 /        | 3| /  
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                                3
        #                                            |x |
        #                4     4                     | 3|          
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #                1     2    
        #                           /                                      \ 2
        #                           | x cos theta + x sin theta + sqrt|x | |
        #                           \ 1             2                | 3|  /
        #                              \_______________  ______________/
        #                                              \/
        #                                             sigma~ 
    
    
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt(abs(xNI[2]));
        
        nablaL = np.zeros(3)
        
        nablaL[0] = 4*xNI[0]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[0];
        nablaL[1] = 4*xNI[1]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[1]; 
        nablaL[2] = 3 * np.abs(xNI[2])**2 * np.sign(xNI[2]) + np.abs(xNI[2])**3 / sigma**3 * 1/np.sqrt(np.abs(xNI[2])) * np.sign(xNI[2]);
    
        theta = 0
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3  
    
        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL
    
    def _kappa(self, xNI): 
        """
        Stabilizing controller for NI-part.

        """
        kappa_val = np.zeros(2)
        
        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])
                         
        zeta_val = self._zeta(xNI)
        
        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )
        
        return kappa_val
    
    def _F(self, xNI, eta, theta):
        """
        Marginal function for NI.

        """
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        
        z = eta - self._kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)
      
    def _Cart2NH(self, coords_Cart): 
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.

        """
        
        xNI = np.zeros(3)
        
        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        
        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )
        
        return xNI
  
    def _NH2ctrl_Cart(self, xNI, uNI): 
        """
        Get control for Cartesian NI from NH coordinates.       

        """

        uCart = np.zeros(2)
        
        uCart[0] = uNI[1] + 1/2 * uNI[0] * ( xNI[2] + xNI[0] * xNI[1] )
        uCart[1] = uNI[0]
        
        return uCart

    def compute_action(self, t, observation):
        """
        Compute sampled action.
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t
            
            xNI = self._Cart2NH( observation ) 
            kappa_val = self._kappa(xNI)
            uNI = self.ctrl_gain * kappa_val
            action = self._NH2ctrl_Cart(xNI, uNI)
            
            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])           
            
            self.action_curr = action
            
            # DEBUG ===================================================================   
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']  
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
            # print(R+table+Bl)
            # /DEBUG ===================================================================            
            
            return action    
    
        else:
            return self.action_curr

    def compute_action_vanila(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobotNI.compute_action`, but without invoking the internal clock.

        """
        
        xNI = self._Cart2NH( observation ) 
        kappa_val = self._kappa(xNI)
        uNI = self.ctrl_gain * kappa_val
        action = self._NH2ctrl_Cart(xNI, uNI)
        
        self.action_curr = action
        
        return action

    def compute_LF(self, observation):
        
        xNI = self._Cart2NH( observation ) 
        
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt( np.abs(xNI[2]) )
        
        return xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma**2


# QCar-specific controllers: independent of Sys3WRobotNI
class CtrlPathTracking:
    """Shared sampled path-tracking controller for rear-axle QCar observations.

    Observations: [x_rear, y_rear, yaw, (optional) measured_speed].
    Actions: [forward_speed_m_s, front_wheel_steering_rad].
    Positive CTE = vehicle to LEFT of forward path tangent; positive heading
    error = path_heading - vehicle_heading; positive steering = turn LEFT.
    For Stanley only, errors are evaluated at the FRONT axle, x_front =
    x_rear + L*cos(yaw), y_front = y_rear + L*sin(yaw).

    All parameters are named and in SI units. `stage_obj_pars` accepts w_e,
    w_heading, w_speed, w_command_speed, w_steering.
    """

    def __init__(self, path, wheelbase=0.256, sampling_time=0.05,
                 target_speed=0.5, min_speed=0.08, max_speed=0.75,
                 max_steering=0.5, max_lateral_acceleration=0.8,
                 curve_speed_safety_factor=0.92, curvature_deadband=0.02,
                 max_accel=1.2, max_decel=1.2, tracking_axle='rear',
                 nearest_lookback_points=12, nearest_lookahead_points=80,
                 path_mode='auto', closed_path_gap_factor=2.5, stage_obj_pars=None,
                 speed_policy='constant'):
        if wheelbase <= 0 or sampling_time <= 0 or max_steering <= 0:
            raise ValueError('wheelbase, sampling_time, max_steering must be > 0')
        if min_speed < 0 or max_speed < min_speed:
            raise ValueError('Invalid speed limits')
        if max_lateral_acceleration <= 0 or max_accel < 0 or max_decel < 0:
            raise ValueError('Invalid acceleration limits')
        if speed_policy not in ('constant', 'adaptive'):
            raise ValueError('speed_policy must be constant or adaptive')
        self.speed_policy = speed_policy
        if tracking_axle not in ('rear', 'front'):
            raise ValueError('tracking_axle must be rear or front')
        if path_mode not in ('auto', 'open', 'closed'):
            raise ValueError('Invalid path mode')
        self.wheelbase = float(wheelbase)
        self.sampling_time = float(sampling_time)
        self.target_speed = float(np.clip(target_speed, min_speed, max_speed))
        self.min_speed, self.max_speed = float(min_speed), float(max_speed)
        self.max_steering = float(max_steering)
        self.max_lateral_acceleration = float(max_lateral_acceleration)
        self.curve_speed_safety_factor = float(curve_speed_safety_factor)
        self.curvature_deadband = float(curvature_deadband)
        self.max_accel, self.max_decel = float(max_accel), float(max_decel)
        self.tracking_axle = tracking_axle
        self.nearest_lookback_points = int(nearest_lookback_points)
        self.nearest_lookahead_points = int(nearest_lookahead_points)
        self.path_mode = path_mode
        self.closed_path_gap_factor = float(closed_path_gap_factor)
        default = dict(w_e=20., w_heading=10., w_speed=2.,
                       w_command_speed=0.0, w_steering=0.1)
        if stage_obj_pars is not None:
            unknown = set(stage_obj_pars) - set(default)
            if unknown:
                raise ValueError(f'Unknown stage objective terms: {unknown}')
            default.update(stage_obj_pars)
        self.stage_obj_pars = default
        self._last_time = None
        self.action_curr = np.zeros(2)
        self.steering_curr = 0.
        self.current_cte = 0.
        self.current_heading_error = 0.
        self.current_path_curvature = 0.
        self.current_velocity = 0.
        self.speed_reference_curr = self.target_speed
        self.accum_obj_val = 0.
        self.current_stage_obj = 0.
        self._nearest_index = None
        self._speed_command = None
        self.last_observation = None
        self.last_sample_computed = False
        self.performance = {}
        self._performance_prev_wall = None
        self._performance_cycle_count = 0
        self._performance_deadline_miss_count = 0
        self.update_path(path)

    @staticmethod
    def _wrap(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    def update_path(self, path):
        """Set a reference path (N x 2 or N x 3). Not a measurement setter."""
        p = np.asarray(path, dtype=float)
        if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] not in (2, 3):
            raise ValueError('Path must be Nx2 or Nx3 with at least 3 points')
        if not np.all(np.isfinite(p)):
            raise ValueError('Path contains nonfinite values')
        xy = p[:, :2].copy()
        spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if np.any(spacing < 1.e-8):
            # A repeated closing endpoint changes none of the geometry.
            keep = np.r_[True, spacing >= 1.e-8]
            xy = xy[keep]
        if len(xy) < 3:
            raise ValueError('Path must have at least 3 unique points')
        # Static ROS path publishers may rebroadcast the same path repeatedly.
        # Rebuilding an identical path would reset _nearest_index and allow
        # figure-eight tracking to jump to the wrong branch at intersections.
        # Preserve the progress hint unless the path geometry actually changes.
        if (hasattr(self, 'path_points') and xy.shape == self.path_points.shape
                and np.allclose(xy, self.path_points, rtol=0.0, atol=1.e-10)):
            return
        gap = float(np.linalg.norm(xy[0]-xy[-1]))
        typical_spacing = float(np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        closed = (self.path_mode == 'closed' or
                  (self.path_mode == 'auto' and gap < self.closed_path_gap_factor*typical_spacing))
        starts = xy if closed else xy[:-1]
        ends = np.roll(xy, -1, axis=0) if closed else xy[1:]
        diff = ends - starts
        lens = np.linalg.norm(diff, axis=1)
        if np.any(lens <= 1.e-8):
            raise ValueError('Path has zero-length segments')
        head = np.arctan2(diff[:, 1], diff[:, 0])
        # Central discrete curvature based on adjacent segment headings.
        dhead = np.asarray([self._wrap(head[(i+1) % len(head)]-head[(i-1) % len(head)])
                            if closed else
                            self._wrap(head[min(i+1,len(head)-1)]-head[max(i-1,0)])
                            for i in range(len(head))])
        denom = np.asarray([lens[(i+1) % len(lens)] + lens[(i-1) % len(lens)]
                            if closed else
                            lens[min(i+1,len(lens)-1)] + lens[max(i-1,0)]
                            for i in range(len(head))])
        self.path_points = xy
        self.path_closed = bool(closed)
        self._seg_start = starts
        self._seg_delta = diff
        self._seg_len = lens
        self._seg_len2 = lens**2
        self._seg_heading = head
        self._seg_curvature = dhead / np.maximum(denom, 1.e-8)
        self._nearest_index = None

    def _geometry(self, state, index_hint=None, tracking_axle=None):
        """Compute nearest SEGMENT projection and signed path errors once.

        `index_hint` is used to maintain path progress at figure-eight crossings.
        Returns dict with continuous projected point, segment index, progress.
        Does not mutate controller state.
        """
        x, y, yaw = map(float, np.asarray(state)[:3])
        axle = self.tracking_axle if tracking_axle is None else tracking_axle
        if axle == 'front':
            x += self.wheelbase*np.cos(yaw)
            y += self.wheelbase*np.sin(yaw)
        pos = np.array([x, y])
        n = len(self._seg_len)
        if index_hint is None:
            indices = np.arange(n)
        elif self.path_closed:
            offsets = np.arange(-self.nearest_lookback_points,
                                self.nearest_lookahead_points+1)
            indices = np.unique((int(index_hint)+offsets) % n)
        else:
            indices = np.arange(max(0, index_hint-self.nearest_lookback_points),
                                min(n, index_hint+self.nearest_lookahead_points+1))
        v = pos - self._seg_start[indices]
        frac = np.clip(np.einsum('ij,ij->i', v, self._seg_delta[indices]) /
                       self._seg_len2[indices], 0., 1.)
        projections = self._seg_start[indices] + frac[:, None]*self._seg_delta[indices]
        distances = np.sum((pos-projections)**2, axis=1)
        j = int(np.argmin(distances))
        idx = int(indices[j])
        unit_tangent = self._seg_delta[idx]/self._seg_len[idx]
        error_vec = pos-projections[j]
        cte = float(unit_tangent[0]*error_vec[1] - unit_tangent[1]*error_vec[0])
        return dict(cte=cte,
                    heading_error=self._wrap(self._seg_heading[idx]-yaw),
                    curvature=float(self._seg_curvature[idx]),
                    index=idx, fraction=float(frac[j]),
                    closest=projections[j])

    def _lookahead(self, geometry, distance):
        """Interpolate a lookahead point along the path from the projection."""
        idx = geometry['index']
        remaining = float(distance) + geometry['fraction']*self._seg_len[idx]
        n = len(self._seg_len)
        for _ in range(n):
            length = self._seg_len[idx]
            if remaining <= length:
                return self._seg_start[idx] + (remaining/length)*self._seg_delta[idx]
            remaining -= length
            if idx == n-1 and not self.path_closed:
                return self.path_points[-1]
            idx = (idx+1) % n
        return self._seg_start[idx]

    def curvature_speed_reference(self, curvature):
        """v_ref = min(target_speed, safety*sqrt(a_lat_max/|curvature|))."""
        if self.speed_policy == 'constant' or abs(curvature) <= self.curvature_deadband:
            return self.target_speed
        curve_speed = (self.curve_speed_safety_factor *
                       np.sqrt(self.max_lateral_acceleration / max(abs(curvature), 1.e-8)))
        return float(np.clip(curve_speed, self.min_speed, self.target_speed))

    def _slew_speed(self, speed_ref, dt, measured_v):
        if self._speed_command is None:
            self._speed_command = float(np.clip(measured_v, self.min_speed, self.max_speed))
        limit = self.max_accel if speed_ref >= self._speed_command else self.max_decel
        self._speed_command += float(np.clip(speed_ref-self._speed_command, -limit*dt, limit*dt))
        return float(np.clip(self._speed_command, self.min_speed, self.max_speed))

    def stage_obj(self, observation, action):
        """One path-tracking stage objective, shared by all controllers."""
        g = self._geometry(observation, self._nearest_index)
        v = float(np.asarray(observation)[3]) if len(observation) > 3 else float(action[0])
        ref = self.curvature_speed_reference(g['curvature'])
        w = self.stage_obj_pars
        return float(w['w_e']*g['cte']**2 +
                     w['w_heading']*g['heading_error']**2 +
                     w['w_speed']*(v-ref)**2 +
                     w['w_command_speed']*float(action[0])**2 +
                     w['w_steering']*float(action[1])**2)

    def upd_accum_obj(self, observation, action, dt=None):
        """Update integrated tracking cost (called ONCE per control cycle)."""
        h = self.sampling_time if dt is None else float(dt)
        self.current_stage_obj = self.stage_obj(observation, action)
        self.accum_obj_val += self.current_stage_obj*h
        return self.accum_obj_val

    def reset(self, t0=None):
        """Reset the episode's progress, controller clock and cost."""
        self._last_time = None
        self._nearest_index = None
        self._speed_command = None
        self.action_curr = np.zeros(2)
        self.steering_curr = 0.
        self.accum_obj_val = 0.
        self.current_stage_obj = 0.
        self.last_observation = None
        self.last_sample_computed = False
        self.performance = {}
        self._performance_prev_wall = None
        self._performance_cycle_count = 0
        self._performance_deadline_miss_count = 0

    def _steering(self, cte, heading_error, measured_v, curvature, geometry, state, dt):
        """Override in concrete controllers. May return steering or (v, steering)."""
        raise NotImplementedError

    def _compute_action_once(self, t, observation):
        """Sole per-cycle controller entrypoint, with no ROS-side error injection."""
        obs = np.asarray(observation, dtype=float).ravel()
        if obs.size not in (3, 4) or not np.all(np.isfinite(obs)):
            raise ValueError('Observation must be finite [x,y,yaw] or [x,y,yaw,v]')
        t = float(t)
        if self._last_time is not None:
            elapsed = t-self._last_time
            if elapsed < -1.e-7:
                raise ValueError('Nonmonotonic controller time; use a single time base')
            if elapsed+1.e-9 < self.sampling_time:
                self.last_sample_computed = False
                return self.action_curr.copy()
        else:
            elapsed = self.sampling_time
        geometry = self._geometry(obs, self._nearest_index)
        self._nearest_index = geometry['index']
        self.current_cte = geometry['cte']
        self.current_heading_error = geometry['heading_error']
        self.current_path_curvature = geometry['curvature']
        self.current_velocity = (float(obs[3]) if len(obs) == 4 else
                                 float(self.action_curr[0]))
        self.speed_reference_curr = self.curvature_speed_reference(geometry['curvature'])
        nominal_speed = (self.target_speed if self.speed_policy == 'constant' else
                         self._slew_speed(self.speed_reference_curr,
                                          elapsed, self.current_velocity))
        result = self._steering(geometry['cte'], geometry['heading_error'],
                                self.current_velocity, geometry['curvature'],
                                geometry, obs, elapsed)
        if isinstance(result, tuple):
            v_cmd, steering = result
        else:
            v_cmd, steering = nominal_speed, result
        action = np.array([np.clip(v_cmd,self.min_speed,self.max_speed),
                           np.clip(steering,-self.max_steering,self.max_steering)], dtype=float)
        self.action_curr = action
        self.steering_curr = float(action[1])
        self._last_time = t
        self.last_observation = obs.copy()
        self.upd_accum_obj(obs, action, elapsed)
        self.last_sample_computed = True
        return action.copy()

    def compute_action(self, t, observation):
        """Timed QCar control call; telemetry belongs to this controller instance.

        Execution time is the *controller* time, not ROS callback + logging.
        Wall-frequency and CPU percentage are meaningful only in a paced
        real-time loop; in fast offline simulation they measure throughput.
        """
        import time
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        action = self._compute_action_once(t, observation)
        if self.last_sample_computed:
            execution_seconds = time.perf_counter() - start_wall
            cpu_seconds = time.process_time() - start_cpu
            previous = getattr(self, '_performance_prev_wall', None)
            interval = (start_wall - previous) if previous is not None else None
            valid = interval is not None and interval > 0
            self._performance_prev_wall = start_wall
            count = getattr(self, '_performance_cycle_count', 0) + 1
            misses = getattr(self, '_performance_deadline_miss_count', 0)
            miss = execution_seconds > self.sampling_time
            misses += int(miss)
            self._performance_cycle_count = count
            self._performance_deadline_miss_count = misses
            self.performance = {
                'controller_loop_frequency_hz': (1.0 / interval if valid else float('nan')),
                'controller_execution_time_ms': execution_seconds * 1000.,
                'controller_cpu_usage_percent': (100. * cpu_seconds / interval
                                                 if valid else float('nan')),
                'controller_deadline_miss': int(miss),
                'controller_deadline_miss_ratio': misses / count,
                'controller_cycle_count': count,
                'controller_deadline_miss_count': misses,
            }
        return action


class CtrlPurePursuit(CtrlPathTracking):
    """Pure Pursuit steering at the REAR axle (standard geometric law)."""

    def __init__(self, path, lookahead_distance=0.5, **kwargs):
        super().__init__(path=path, tracking_axle='rear', **kwargs)
        if lookahead_distance <= 0:
            raise ValueError('lookahead_distance must be positive')
        self.lookahead_distance = float(lookahead_distance)

    def _steering(self, cte, heading_error, measured_v, curvature, geometry, state, dt):
        target = self._lookahead(geometry, self.lookahead_distance)
        dx, dy = target - state[:2]
        dist = float(np.hypot(dx, dy))
        if dist < 1.e-6:
            return 0.0
        alpha = self._wrap(np.arctan2(dy,dx)-state[2])
        return float(np.arctan2(2*self.wheelbase*np.sin(alpha), dist))


class CtrlStanley(CtrlPathTracking):
    """Stanley at FRONT axle. δ = heading_error - atan2(k*cte, v+softening)."""

    def __init__(self, path, k=0.8, softening_gain=0.1,
                 min_velocity=0.1, steering_sign=1., **kwargs):
        super().__init__(path=path, tracking_axle='front', **kwargs)
        self.k = float(k)
        self.softening_gain = float(softening_gain)
        self.min_velocity = float(min_velocity)
        self.steering_sign = float(steering_sign)

    def _steering(self, cte, heading_error, measured_v, curvature, geometry, state, dt):
        velocity = max(abs(measured_v),self.min_velocity)
        return self.steering_sign*(heading_error -
                                   np.arctan2(self.k*cte,velocity+self.softening_gain))


class CtrlPID(CtrlPathTracking):
    """CTE PID at REAR axle, filtered derivative and saturation anti-windup."""

    def __init__(self, path, kp=0.85, ki=0.005, kd=0.32,
                 heading_gain=0.0, curvature_feedforward_gain=1.0, integral_limit=0.5,
                 derivative_filter_tau=0.12, max_dt=0.2,
                 steering_sign=1., **kwargs):
        super().__init__(path=path, tracking_axle='rear', **kwargs)
        self.kp,self.ki,self.kd = float(kp),float(ki),float(kd)
        self.heading_gain = float(heading_gain)
        self.curvature_feedforward_gain = float(curvature_feedforward_gain)
        self.integral_limit = float(integral_limit)
        self.derivative_filter_tau = float(derivative_filter_tau)
        self.max_dt = float(max_dt)
        self.steering_sign = float(steering_sign)
        self._prev_cte = None
        self._integral = 0.
        self._filtered_derivative = 0.

    def reset(self, t0=None):
        super().reset(t0)
        self._prev_cte = None
        self._integral = 0.
        self._filtered_derivative = 0.

    def _steering(self, cte, heading_error, measured_v, curvature, geometry, state, dt):
        h = min(dt, self.max_dt)
        raw_derivative = 0. if self._prev_cte is None else (cte-self._prev_cte)/h
        alpha = (self.derivative_filter_tau/(self.derivative_filter_tau+h)
                 if self.derivative_filter_tau > 0 else 0.)
        derivative = alpha*self._filtered_derivative+(1-alpha)*raw_derivative
        candidate = float(np.clip(self._integral+cte*h,
                                  -self.integral_limit, self.integral_limit))
        def raw(integral):
            return self.steering_sign*(
                       self.curvature_feedforward_gain*np.arctan(self.wheelbase*curvature)+
                       self.heading_gain*heading_error-
                       self.kp*cte-self.ki*integral-self.kd*derivative)
        steering = raw(candidate)
        # Don't integrate if integration would push a saturated command farther out.
        integral_effect = -self.steering_sign*self.ki*cte*h
        if ((steering > self.max_steering and integral_effect > 0.) or
            (steering < -self.max_steering and integral_effect < 0.)):
            candidate = self._integral
            steering = raw(candidate)
        self._integral = candidate
        self._prev_cte = cte
        self._filtered_derivative = derivative
        return steering


class CtrlMPC(CtrlPathTracking):
    """Model-based nonlinear receding-horizon joint (steer,accel) QCar MPC.

    Uses sys_rhs = my_sys._state_dyn EXACTLY for each prediction step, with
    receding-horizon Nactor and pred_step_size. Optimizer: scipy L-BFGS-B,
    numerical gradients. This is a model-based replacement for the previous
    error-coordinate adjoint solver, not a claim of equivalent performance.
    """

    def __init__(self, path, sys_rhs, Nactor=15, pred_step_size=0.05,
                 optimizer_max_iterations=25, max_acceleration=0.8,
                 max_deceleration=1.2, max_steering_rate=2.,
                 w_accel=0.2, w_steering_rate=0.08,
                 terminal_weight=3., w_steering_rate_violation=200.,
                 w_speed_overshoot=250.,
                 optimizer_ftol=1.e-5, **kwargs):
        # May be enabled ONLY for ROS camera mode. A fixed Gazebo reference
        # continues to use the original warm-start and path-progress logic.
        self.dynamic_path_mode = False
        self.dynamic_path_cte_reset_m = 0.10
        self.dynamic_path_heading_reset_rad = 0.30
        self.dynamic_path_reset_count = 0
        # Only enabled for ROS camera-derived OPEN paths. Fixed reference paths
        # retain bit-for-bit identical objective and warm-start behavior.
        self.finite_path_mode = False
        self.finite_path_end_buffer_m = 0.15
        self.finite_path_brake_decel_mps2 = 0.40
        self.finite_path_overrun_weight = 60.0
        super().__init__(path=path, tracking_axle='rear', **kwargs)
        if not callable(sys_rhs):
            raise ValueError('sys_rhs must be my_sys._state_dyn')
        if Nactor < 2 or pred_step_size <= 0:
            raise ValueError('Nactor>=2 and pred_step_size>0 required')
        self.sys_rhs = sys_rhs
        self.Nactor = int(Nactor)
        self.pred_step_size = float(pred_step_size)
        self.optimizer_max_iterations = int(optimizer_max_iterations)
        self.max_acceleration = float(max_acceleration)
        self.max_deceleration = float(max_deceleration)
        self.max_steering_rate = float(max_steering_rate)
        self.w_accel = float(w_accel)
        self.w_steering_rate = float(w_steering_rate)
        self.terminal_weight = float(terminal_weight)
        self.w_steering_rate_violation = float(w_steering_rate_violation)
        # Asymmetric cost: matching v_ref is a preference; exceeding the
        # curvature-informed envelope is specifically discouraged.
        self.w_speed_overshoot = float(w_speed_overshoot)
        if self.w_speed_overshoot < 0:
            raise ValueError('w_speed_overshoot must be nonnegative')
        self.optimizer_ftol = float(optimizer_ftol)
        self._last_solution = None
        self.optimizer_result = None

    def reset(self, t0=None):
        super().reset(t0)
        self._last_solution = None
        self.optimizer_result = None

    def set_dynamic_path_mode(self, enabled=True):
        """Enable dynamic camera-path adaptation without altering MPC control law."""
        self.dynamic_path_mode = bool(enabled)
        self._last_solution = None  # never import a static-reference warm start

    def set_finite_path_mode(self, enabled=True, *, end_buffer_m=0.15,
                             brake_decel_mps2=0.40, overrun_weight=60.0):
        """Limit MPC planning to the measured part of a moving local path.

        The ROS adapter enables this only for MPC+lane. No virtual waypoints
        are appended and reference-trajectory mode remains unchanged.
        """
        if end_buffer_m < 0 or brake_decel_mps2 <= 0 or overrun_weight < 0:
            raise ValueError('Invalid finite-path parameters')
        self.finite_path_mode = bool(enabled)
        self.finite_path_end_buffer_m = float(end_buffer_m)
        self.finite_path_brake_decel_mps2 = float(brake_decel_mps2)
        self.finite_path_overrun_weight = float(overrun_weight)
        self._last_solution = None

    def _remaining_preview_m(self, geometry):
        """Observed arc length remaining from nearest segment projection.

        Conservative if the QCar is still behind the camera's first waypoint:
        we do NOT add an unobserved segment from the axle to the camera.
        """
        idx = int(geometry['index'])
        frac = float(geometry.get('fraction', 0.0))
        return max(0.0, float(np.sum(self._seg_len[idx+1:]) +
                              (1.0-frac)*self._seg_len[idx]))

    def _beyond_observed_path(self, geometry, state):
        """Signed forward overrun past an OPEN path's last measured point.

        A position laterally offset from the endpoint is NOT automatically an
        overrun. For stationary reference trajectories this returns zero.
        """
        if not self.finite_path_mode or self.path_closed:
            return 0.0
        if int(geometry['index']) != len(self._seg_len)-1:
            return 0.0
        tangent = self._seg_delta[-1] / self._seg_len[-1]
        return max(0.0, float(np.dot(np.asarray(state)[:2] -
                                      self.path_points[-1], tangent)))

    def update_path(self, path):
        # Static-mode behavior is EXACTLY the existing base implementation.
        if not getattr(self, 'dynamic_path_mode', False):
            return super().update_path(path)
        old_geom = None
        if self.last_observation is not None:
            old_geom = self._geometry(self.last_observation, self._nearest_index)
        previous = getattr(self, 'path_points', None)
        super().update_path(path)
        # Do not carry an index from a DIFFERENT moving local camera path.
        if previous is not None and (previous.shape != self.path_points.shape or
                                      not np.array_equal(previous, self.path_points)):
            self._nearest_index = None
        if old_geom is not None:
            new_geom = self._geometry(self.last_observation)
            error_jump = abs(new_geom['cte'] - old_geom['cte'])
            heading_jump = abs(self._wrap(new_geom['heading_error'] - old_geom['heading_error']))
            if (error_jump > self.dynamic_path_cte_reset_m or
                    heading_jump > self.dynamic_path_heading_reset_rad):
                self._last_solution = None
                self.dynamic_path_reset_count += 1

    def _preview_speed_reference(self, geom):
        """Speed envelope from *ordered reference-path* curvature, not vehicle yaw.

        A bend ahead has allowable speed v_curve; a vehicle can travel faster
        now only if it has enough distance to decelerate to that speed:
            v_now <= sqrt(v_curve**2 + 2*a_decel*distance_to_bend).
        Following waypoint order disambiguates the figure-eight crossing.
        This function does not mutate path progress or controller state.
        """
        if self.speed_policy == 'constant':
            return self.target_speed
        idx = int(geom['index'])
        fraction = float(geom.get('fraction', 0.))
        limit = self.curvature_speed_reference(self._seg_curvature[idx])
        if self.max_deceleration <= 0.:
            return float(limit)
        # Preview at least the MPC horizon plus enough stopping distance.
        stopping_distance = max(0., self.target_speed**2 - self.min_speed**2) / (2*self.max_deceleration)
        preview = max(self.target_speed*self.Nactor*self.pred_step_size,
                      stopping_distance, 0.25)
        distance = (1.-fraction)*self._seg_len[idx]
        n = len(self._seg_len)
        for _ in range(n-1):
            if distance > preview:
                break
            idx += 1
            if idx >= n:
                if not self.path_closed:
                    break
                idx = 0
            curve_limit = self.curvature_speed_reference(self._seg_curvature[idx])
            braking_limit = np.sqrt(curve_limit**2 + 2*self.max_deceleration*distance)
            limit = min(limit, braking_limit)
            distance += self._seg_len[idx]
        if self.finite_path_mode and not self.path_closed:
            # Brake toward the END OF THE *observed* path; do not hallucinate
            # straight road beyond it. Actual zero-stop is enforced by ROS.
            remaining = self._remaining_preview_m(geom)
            braking_length = max(0.0, remaining - self.finite_path_end_buffer_m)
            limit = min(limit, max(self.min_speed,
                np.sqrt(2.0*self.finite_path_brake_decel_mps2*braking_length)))
        return float(np.clip(limit, self.min_speed, self.target_speed))

    def _rollout_and_gradient(self, x0, v0, sequence, progress_index,
                              compute_gradient=True):
        """Predict using SysQCar RHS, and optionally compute discrete adjoint.

        Gradient is exact for the midpoint kinematic bicycle integration within
        each current path segment and for unsaturated velocity. At segment
        transitions, speed clipping and rate-penalty boundaries, it is piecewise
        differentiable. Finite-difference tests cover interior cases.
        """
        h = self.pred_step_size
        state = np.asarray(x0[:3], dtype=float).copy()
        velocity = float(np.clip(v0, self.min_speed, self.max_speed))
        prev_steer = float(self.action_curr[1])
        cost = 0.
        idx = progress_index
        jacobians = []
        gradients_state = []
        gradients_input = []
        path_states = [state.copy()]
        for k in range(self.Nactor):
            accel = float(sequence[2*k])
            steer = float(sequence[2*k+1])
            unclipped_speed = velocity + accel*h
            # Cruise speed is a HARD upper bound, not just a soft MPC weight.
            # This avoids the optimizer speeding up above target_speed to
            # reduce cross-track costs on sharp figure-eight bends.
            v_max = (self.target_speed if self.speed_policy == 'adaptive'
                     else self.max_speed)
            v_next = float(np.clip(unclipped_speed, self.min_speed, v_max))
            velocity_derivative = float(self.min_speed <= unclipped_speed < v_max)
            v_for_model = 0.5*(velocity+v_next)
            # Call the configured system dynamics on BOTH RK2 evaluations.
            # No independently integrated error-coordinate motion model.
            action = np.array([v_for_model,steer])
            rhs = np.asarray(self.sys_rhs(k*h, state, action, ()),dtype=float)
            midpoint = state + (h/2)*rhs
            rhs_mid = np.asarray(self.sys_rhs((k+0.5)*h,midpoint,action,()),dtype=float)
            next_state = state+h*rhs_mid
            next_state[2]=self._wrap(next_state[2])
            geom = self._geometry(next_state, idx)
            idx = geom['index']
            # Camera observations are finite. Past their endpoint, the last
            # segment is a boundary, NOT an indefinitely extended target lane.
            overrun = self._beyond_observed_path(geom, next_state)
            observed = (overrun <= 1.e-10)
            tracking_weight = (1.0 if observed else 0.0)
            # Brake proactively for UPCOMING path curvature, not merely the
            # curvature of a possibly off-path predicted vehicle position.
            vref = self._preview_speed_reference(geom)
            excess_speed = max(0., v_next-vref)
            w = self.stage_obj_pars
            steer_diff = steer-prev_steer
            rate_excess=max(0.,abs(steer_diff)/h-self.max_steering_rate)
            rate_penalty = self.w_steering_rate_violation*rate_excess**2
            stage = (tracking_weight*(w['w_e']*geom['cte']**2+
                                     w['w_heading']*geom['heading_error']**2)+
                     self.finite_path_overrun_weight*overrun**2+
                     w['w_speed']*(v_next-vref)**2+
                     self.w_speed_overshoot*excess_speed**2+
                     w['w_command_speed']*v_next**2+
                     w['w_steering']*steer**2+
                     self.w_accel*accel**2+
                     self.w_steering_rate*(steer_diff/h)**2+rate_penalty)
            cost += h*stage
            if compute_gradient:
                # Cost derivatives at z_{k+1}=[x,y,yaw,v] and u=[a,delta].
                tangent=self._seg_delta[idx]/self._seg_len[idx]
                boundary_tangent = self._seg_delta[-1]/self._seg_len[-1]
                boundary_grad = 2*self.finite_path_overrun_weight*overrun*boundary_tangent
                grad_z=np.array([
                    tracking_weight*(-2*w['w_e']*geom['cte']*tangent[1]) + boundary_grad[0],
                    tracking_weight*(2*w['w_e']*geom['cte']*tangent[0]) + boundary_grad[1],
                    tracking_weight*(-2*w['w_heading']*geom['heading_error']),
                    2*w['w_speed']*(v_next-vref)+
                    2*self.w_speed_overshoot*excess_speed+
                    2*w['w_command_speed']*v_next
                ])
                grad_u=np.array([2*self.w_accel*accel,
                                 2*w['w_steering']*steer])
                # Jacobian of the midpoint-discretized QCar dynamics.
                # Yaw derivative uses SysQCar wheelbase; model supplied via sys_rhs.
                g=np.tan(steer)/self.wheelbase
                dg=1.0/(self.wheelbase*np.cos(steer)**2)
                mid_yaw=state[2]+0.5*h*v_for_model*g
                co,si=np.cos(mid_yaw),np.sin(mid_yaw)
                dv_dv=0.5*(1+velocity_derivative)
                dv_da=0.5*h*velocity_derivative
                v_x=h*(co-0.5*h*v_for_model*si*g)
                v_y=h*(si+0.5*h*v_for_model*co*g)
                A=np.eye(4)
                A[0,2]=-h*v_for_model*si
                A[1,2]=h*v_for_model*co
                A[0,3]=v_x*dv_dv
                A[1,3]=v_y*dv_dv
                A[2,3]=h*g*dv_dv
                A[3,3]=velocity_derivative
                B=np.zeros((4,2))
                B[0,0]=v_x*dv_da
                B[1,0]=v_y*dv_da
                B[2,0]=h*g*dv_da
                B[3,0]=h*velocity_derivative
                B[0,1]=-0.5*h*h*v_for_model*v_for_model*si*dg
                B[1,1]=0.5*h*h*v_for_model*v_for_model*co*dg
                B[2,1]=h*v_for_model*dg
                jacobians.append((A,B))
                gradients_state.append(grad_z)
                gradients_input.append(grad_u)
            state=next_state
            velocity=v_next
            prev_steer=steer
            path_states.append(state.copy())
        terminal=self._geometry(state,idx)
        terminal_overrun = self._beyond_observed_path(terminal, state)
        terminal_tracking = (1.0 if terminal_overrun <= 1.e-10 else 0.0)
        cost += self.terminal_weight*(
            terminal_tracking*(self.stage_obj_pars['w_e']*terminal['cte']**2+
                               self.stage_obj_pars['w_heading']*terminal['heading_error']**2)+
            self.finite_path_overrun_weight*terminal_overrun**2)
        if not compute_gradient:
            return float(cost), None, np.asarray(path_states)
        tangent=self._seg_delta[terminal['index']]/self._seg_len[terminal['index']]
        boundary_tangent = self._seg_delta[-1]/self._seg_len[-1]
        terminal_boundary_grad = (2*self.terminal_weight*self.finite_path_overrun_weight*
                                  terminal_overrun*boundary_tangent)
        lam=np.array([
            terminal_tracking*(-2*self.terminal_weight*self.stage_obj_pars['w_e']*terminal['cte']*tangent[1]) + terminal_boundary_grad[0],
            terminal_tracking*(2*self.terminal_weight*self.stage_obj_pars['w_e']*terminal['cte']*tangent[0]) + terminal_boundary_grad[1],
            terminal_tracking*(-2*self.terminal_weight*self.stage_obj_pars['w_heading']*terminal['heading_error']),
            0.,
        ])
        grad=np.zeros(2*self.Nactor)
        for k in range(self.Nactor-1,-1,-1):
            A,B=jacobians[k]
            lam_next=lam+h*gradients_state[k]
            grad[2*k:2*k+2] = B.T@lam_next+h*gradients_input[k]
            lam=A.T@lam_next
        # Steering-rate coupling between successive candidate decisions.
        prev_delta=float(self.action_curr[1])
        for k in range(self.Nactor):
            delta=float(sequence[2*k+1])
            diff=delta-prev_delta
            rate_grad=2*self.w_steering_rate*diff/h
            rate_excess=max(0.,abs(diff)/h-self.max_steering_rate)
            if rate_excess>0:
                rate_grad+=2*self.w_steering_rate_violation*rate_excess*np.sign(diff)
            grad[2*k+1]+=rate_grad
            if k>0:
                grad[2*(k-1)+1]-=rate_grad
            prev_delta=delta
        return float(cost),grad,np.asarray(path_states)

    def _rollout(self, x0, v0, sequence, progress_index):
        """Pure prediction cost and trajectory (retained for diagnostics)."""
        cost,_,states=self._rollout_and_gradient(x0,v0,sequence,
                                                 progress_index,False)
        return cost,states

    def _solve(self, state, speed, curvature, progress_index):
        """Warm-start L-BFGS-B with model-based discrete adjoint, not random."""
        from scipy.optimize import minimize
        n = self.Nactor
        ff = float(np.clip(np.arctan(self.wheelbase*curvature),
                           -self.max_steering,self.max_steering))
        if self._last_solution is None:
            seq = np.tile([0.,ff],n)
        else:
            seq = np.concatenate([self._last_solution[2:],self._last_solution[-2:]])
        accel_bounds = ((0., 0.) if self.speed_policy == 'constant'
                        else (-self.max_deceleration, self.max_acceleration))
        bounds = [accel_bounds, (-self.max_steering,self.max_steering)]*n
        memo={'x':None,'cost':None,'grad':None}
        def objective(z):
            if memo['x'] is None or not np.array_equal(z,memo['x']):
                memo['cost'],memo['grad'],_=self._rollout_and_gradient(
                    state,speed,z,progress_index,True)
                memo['x']=z.copy()
            return memo['cost'],memo['grad']
        result = minimize(objective,seq,method='L-BFGS-B',jac=True,bounds=bounds,
                          options=dict(maxiter=self.optimizer_max_iterations,ftol=self.optimizer_ftol))
        sequence=np.asarray(result.x,dtype=float) if np.all(np.isfinite(result.x)) else seq
        if self.speed_policy == 'constant':
            sequence[::2] = 0.0
        self._last_solution=sequence.copy()
        self.optimizer_result=dict(success=bool(result.success),
                                   iterations=int(result.nit),
                                   cost=float(objective(sequence)[0]))
        return sequence

    def _steering(self, cte, heading_error, measured_v, curvature, geometry, state, dt):
        speed = float(np.clip(measured_v if measured_v>0 else
                              (self.action_curr[0] if self.action_curr[0]>0 else
                               self.min_speed), self.min_speed,self.max_speed))
        prediction_speed = (self.target_speed if self.speed_policy == 'constant' else speed)
        seq = self._solve(state,prediction_speed,curvature,geometry['index'])
        acceleration,steering = float(seq[0]),float(seq[1])
        if self.speed_policy == 'constant':
            commanded_speed = self.target_speed
        else:
            preview_ref = self._preview_speed_reference(geometry)
            self.speed_reference_curr = preview_ref
            # The MPC still optimizes acceleration, but may not command speed
            # above the curvature-derived envelope or the cruise target.
            optimizer_speed = float(np.clip(speed + acceleration*dt,
                                            self.min_speed, self.target_speed))
            desired = min(optimizer_speed, preview_ref)
            # Respect the maximum deceleration per command interval. If a
            # corner begins inside this braking distance, speed can temporarily
            # exceed the envelope; the preview is meant to avoid that case.
            reachable_min = max(self.min_speed, speed-self.max_deceleration*dt)
            commanded_speed = float(np.clip(max(desired, reachable_min),
                                             self.min_speed, self.target_speed))
        # Bound instantaneous steering change, consistent with optimizer penalty.
        steering = float(np.clip(steering,
                        self.action_curr[1]-self.max_steering_rate*dt,
                        self.action_curr[1]+self.max_steering_rate*dt))
        self._speed_command = commanded_speed
        return commanded_speed,steering
