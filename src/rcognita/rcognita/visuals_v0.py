#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import numpy.linalg as la
from .utilities import upd_line
from .utilities import reset_line
from .utilities import upd_scatter
from .utilities import upd_text
from .utilities import to_col_vec

import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from mpldatacursor import datacursor
except ImportError:
    datacursor = None

# !pip install svgpath2mpl matplotlib <-- to install this
try:
    from svgpath2mpl import parse_path
except ImportError:
    parse_path = None

from collections import namedtuple

class Animator:
    """
    Interface class of visualization machinery for simulation of system-controller loops.
    To design a concrete animator: inherit this class, override:
        | :func:`~visuals.Animator.__init__` :
        | define necessary visual elements (required)
        | :func:`~visuals.Animator.init_anim` :
        | initialize necessary visual elements (required)        
        | :func:`~visuals.Animator.animate` :
        | animate visual elements (required)
    
    Attributes
    ----------
    objects : : tuple
        Objects to be updated within animation cycle
    pars : : tuple
        Fixed parameters of objects and visual elements      
    
    """
    def __init__(self, objects=[], pars=[]):
        pass
    
    def init_anim(self):
        pass
     
    def animate(self, k):
        pass
    
    def get_anm(self, anm):
        """
        ``anm`` should be a ``FuncAnimation`` object.
        This method is needed to hand the animator access to the currently running animation, say, via ``anm.event_source.stop()``.
    
        """   
        self.anm = anm
        
    def stop_anm(self):
        """
        Stops animation, provided that ``self.anm`` was defined via ``get_anm``.

        """
        self.anm.event_source.stop()           
        # plt.close('all')
        raise Exception('exit')        

class RobotMarker:
    """
    Robot marker for visualization.
    
    """    
    def __init__(self, angle=None, path_string=None):
        #self.angle = angle or []
        self.angle = angle if angle is not None else 0
        self.path_string = path_string or """m 66.893258,227.10128 h 5.37899 v 0.91881 h 1.65571 l 1e-5,-3.8513 3.68556,-1e-5 v -1.43933
        l -2.23863,10e-6 v -2.73937 l 5.379,-1e-5 v 2.73938 h -2.23862 v 1.43933 h 3.68556 v 8.60486 l -3.68556,1e-5 v 1.43158
        h 2.23862 v 2.73989 h -5.37899 l -1e-5,-2.73989 h 2.23863 v -1.43159 h -3.68556 v -3.8513 h -1.65573 l 1e-5,0.91881 h -5.379 z"""
        self.path = parse_path( self.path_string )
        self.path.vertices -= self.path.vertices.mean( axis=0 )
        self.marker = mpl.markers.MarkerStyle( marker=self.path )
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(angle-self.angle)
        self.angle = angle
    
class Animator3WRobot(Animator):
    """
    Animator class for a 3-wheel robot with dynamic actuators. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        state_init, \
        action_init, \
        t0,  \
        t1, \
        state_full_init, \
        xMin, \
        xMax, \
        yMin, \
        yMax, \
        ctrl_mode, \
        action_manual, \
        Fmin, \
        Mmin, \
        Fmax, \
        Mmax, \
        Nruns, \
        is_print_sim_step, \
        is_log_data, \
        is_playback, \
        stage_obj_init = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.state_full_init = state_full_init
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        xCoord0 = state_init[0]
        yCoord0 = state_init[1]
        alpha0 = state_init[2]
        alpha_deg0 = alpha0/2/np.pi
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))    
            
        # xy plane  
        self.axs_xy_plane = self.fig_sim.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax),
                                                  xlabel='x [m]', ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')
        self.axs_xy_plane.set_aspect('equal', adjustable='box')
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
        self.line_traj, = self.axs_xy_plane.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = 't = {time:2.3f}'.format(time = t0)
        self.text_time_handle = self.axs_xy_plane.text(0.05, 0.95, text_time,
                                                   horizontalalignment='left', verticalalignment='center', transform=self.axs_xy_plane.transAxes)
        self.axs_xy_plane.format_coord = lambda state,observation: '%2.2f, %2.2f' % (state,observation)
        
        # Solution
        self.axs_sol = self.fig_sim.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_norm, = self.axs_sol.plot(t0, la.norm([xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.line_alpha, = self.axs_sol.plot(t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
        self.axs_sol.legend(fancybox=True, loc='upper right')
        self.axs_sol.format_coord = lambda state,observation: '%2.2f, %2.2f' % (state,observation)
        
        # Cost
        if is_playback:
            stage_obj = stage_obj_init
        else:
            observation_init = self.sys.out(state_init)
            stage_obj = self.ctrl_benchmarking.stage_obj(observation_init, action_init)
        
        self.axs_cost = self.fig_sim.add_subplot(223, autoscale_on=False, xlim=(t0,t1), ylim=(0, 1e4*stage_obj), yscale='symlog', xlabel='t [s]')
        
        text_accum_obj = r'$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.3f}'.format(accum_obj = 0)
        self.text_accum_obj_handle = self.fig_sim.text(0.05, 0.5, text_accum_obj, horizontalalignment='left', verticalalignment='center')
        self.line_heading_error, = self.axs_cost.plot(
            t0, 0, 'b-', lw=0.5,
            label='Heading Error [rad]'
        )
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Control
        self.axs_ctrl = self.fig_sim.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.lines_ctrl = self.axs_ctrl.plot(t0, to_col_vec(action_init).T, lw=0.5)
        self.axs_ctrl.legend(iter(self.lines_ctrl), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['line_traj', 'line_norm', 'line_alpha', 'line_stage_obj', 'line_accum_obj', 'lines_ctrl'])
        self.lines = cLines(line_traj=self.line_traj,
                            line_norm=self.line_norm,
                            line_alpha=self.line_alpha,
                            line_stage_obj=self.line_stage_obj,
                            line_accum_obj=self.line_accum_obj,
                            lines_ctrl=self.lines_ctrl)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)
    
    def set_sim_data(self, ts, xCoords, yCoords, alphas, vs, omegas, rs, accum_objs, Fs, Ms):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.xCoords, self.yCoords, self.alphas, self.vs, self.omegas = ts, xCoords, yCoords, alphas, vs, omegas
        self.rs, self.accum_objs, self.Fs, self.Ms = rs, accum_objs, Fs, Ms
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.state_full = np.array([self.xCoords[self.curr_step], self.yCoords[self.curr_step], self.alphas[self.curr_step], self.vs[self.curr_step], self.omegas[self.curr_step]])
        self.stage_obj = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.Fs[self.curr_step], self.Ms[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        state_init, *_ = self.pars
        
        xCoord0 = state_init[0]
        yCoord0 = state_init[1]       
        
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c='b')
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            state_full = self.state_full
            action = self.action
            stage_obj = self.stage_obj
            accum_obj = self.accum_obj        
            
        else:
            self.simulator.sim_step()
            
            t, state, observation, state_full = self.simulator.get_sim_step_data()
            
            action = self.ctrl_selector(t, observation, self.action_manual, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(action)
            self.ctrl_benchmarking.receive_sys_state(self.sys._state) 
            self.ctrl_benchmarking.upd_accum_obj(observation, action)
            
            stage_obj = self.ctrl_benchmarking.stage_obj(observation, action)
            accum_obj = self.ctrl_benchmarking.accum_obj_val
        
        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha/np.pi*180
        v = state_full[3]
        omega = state_full[4]

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action)
        
        # xy plane  
        text_time = 't = {time:2.3f}'.format(time = t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot
            
        self.robot_marker.rotate(1e-3)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord, yCoord, marker=self.robot_marker.marker, s=400, c='b')
        
        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord, yCoord, marker=self.robot_marker.marker, s=400, c='b')
        
        # # Solution
        upd_line(self.line_norm, t, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, t, alpha)
    
        # Cost
        upd_line(self.line_stage_obj, t, stage_obj)
        upd_line(self.line_accum_obj, t, accum_obj)
        text_accum_obj = r'$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}'.format(accum_obj = accum_obj)
        upd_text(self.text_accum_obj_handle, text_accum_obj)
        
        # Control
        for (line, action_single) in zip(self.lines_ctrl, action):
            upd_line(line, t, action_single)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))
                
            self.run_curr += 1
            
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            accum_obj = 0     
            
            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_stage_obj)
            reset_line(self.line_accum_obj)
            reset_line(self.lines_ctrl[0])
            reset_line(self.lines_ctrl[1])
            
            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)
    
            upd_line(self.line_traj, np.nan, np.nan)

class Animator3WRobotNI(Animator):
    """
    Animator class for a 3-wheel robot with static actuators. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        state_init, \
        action_init, \
        t0, \
        t1, \
        state_full_init, \
        xMin, \
        xMax, \
        yMin, \
        yMax, \
        ctrl_mode, \
        action_manual, \
        v_min, \
        omega_min, \
        v_max, \
        omega_max, \
        Nruns, \
        is_print_sim_step, \
        is_log_data, \
        is_playback, \
        stage_obj_init = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.state_full_init = state_full_init
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        xCoord0 = state_init[0]
        yCoord0 = state_init[1]
        alpha0 = state_init[2]
        alpha_deg0 = alpha0/2/np.pi
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))   
        gs = self.fig_sim.add_gridspec(2, 3, left=0.06, right=0.98, bottom=0.08, top=0.95, wspace=0.32, hspace=0.35) 
            
        # xy plane  
        self.axs_xy_plane = self.fig_sim.add_subplot(gs[0, 0], autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax),
                                                  xlabel='x [m]', ylabel='y [m]', title='q - quit, click - data cursor')
        self.axs_xy_plane.set_aspect('equal', adjustable='box')
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
        self.line_traj, = self.axs_xy_plane.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = 't = {time:2.3f}'.format(time = t0)
        self.text_time_handle = self.axs_xy_plane.text(0.05, 0.95, text_time,
                                                   horizontalalignment='left', verticalalignment='center', transform=self.axs_xy_plane.transAxes)
        self.axs_xy_plane.format_coord = lambda state,observation: '%2.2f, %2.2f' % (state,observation)
        
        # Solution
        #self.axs_sol = self.fig_sim.add_subplot(gs[0, 1], autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
        self.axs_sol = self.fig_sim.add_subplot(gs[0, 1], autoscale_on=False, xlim=(t0,t1), ylim=(-0.50, 0.50), xlabel='t [s]')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        #-------------updated for qcar---------------------------------------------------------------------------------------------------------------------------------------------------
        
        self.line_cte, = self.axs_sol.plot(t0, 0, 'b-', lw=0.5, label='CTE [m]')
        self.axs_sol.legend(fancybox=True, loc='upper right')
        
        self.axs_sol.format_coord = lambda state,observation: '%2.2f, %2.2f' % (state,observation)
        
        # Stage objective
        self.axs_stage_obj = self.fig_sim.add_subplot(gs[0, 2], autoscale_on=False, xlim=(t0, t1), ylim=(0, 1e6), yscale='symlog', xlabel='t [s]')

        self.line_stage_obj, = self.axs_stage_obj.plot(t0, 0, 'r-', lw=0.5, label='Stage obj.')

        self.line_accum_obj, = self.axs_stage_obj.plot(t0, 0, 'g-', lw=0.5, label=r'$\int \mathrm{Stage\,obj.}\,\mathrm{d}t$')

        self.axs_stage_obj.legend(fancybox=True, loc='upper right')
        
        # Velocity
        self.axs_velocity = self.fig_sim.add_subplot(gs[1, 2], autoscale_on=False, xlim=(t0, t1), ylim=(0, 1.2), xlabel='t [s]', ylabel='Velocity [m/s]')

        self.line_velocity, = self.axs_velocity.plot(t0, 0, 'b-', lw=0.5, label='Velocity')

        self.axs_velocity.legend(fancybox=True, loc='upper right')
        

        # Heading error
        self.axs_cost = self.fig_sim.add_subplot(gs[1, 0],autoscale_on=False,xlim=(t0, t1),ylim=(-0.50, 0.50),xlabel='t [s]')
        self.axs_cost.plot([t0, t1], [0, 0], 'k--', lw=0.75)
        self.line_heading_error, = self.axs_cost.plot(t0, 0, 'b-', lw=0.5, label='Heading error [rad]')
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # steering
        self.axs_ctrl = self.fig_sim.add_subplot(gs[1, 1], autoscale_on=False, xlim=(t0,t1), ylim=(-0.50, 0.50), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_steering, = self.axs_ctrl.plot(t0, 0, 'b-', lw=0.5, label='Steering [rad]')

        self.axs_ctrl.legend(fancybox=True,loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines',['line_traj', 'line_cte', 'line_stage_obj', 'line_accum_obj', 'line_heading_error', 'line_steering', 'line_velocity'])

        self.lines = cLines(
            line_traj=self.line_traj,
            line_cte=self.line_cte,
            line_stage_obj=self.line_stage_obj,
            line_accum_obj=self.line_accum_obj,
            line_heading_error=self.line_heading_error,
            line_steering=self.line_steering,
            line_velocity=self.line_velocity,
        )
        self.times = []
        self.cte_values = []
        self.heading_error_values = []
        self.steering_values = []
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)
    
    def set_sim_data(self, ts, xCoords, yCoords, alphas, rs, accum_objs, vs, omegas):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.xCoords, self.yCoords, self.alphas = ts, xCoords, yCoords, alphas
        self.rs, self.accum_objs, self.vs, self.omegas = rs, accum_objs, vs, omegas
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.state_full = np.array([self.xCoords[self.curr_step], self.yCoords[self.curr_step], self.alphas[self.curr_step]])
        self.stage_obj = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.vs[self.curr_step], self.omegas[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        state_init, *_ = self.pars
        
        xCoord0 = state_init[0]
        yCoord0 = state_init[1]       
        
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c='b')
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            state_full = self.state_full
            action = self.action
            stage_obj = self.stage_obj
            accum_obj = self.accum_obj        
            
        else:
            self.simulator.sim_step()
            
            t, state, observation, state_full = self.simulator.get_sim_step_data()
            
            action = self.ctrl_selector(t, observation, self.action_manual, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(action)
            self.ctrl_benchmarking.receive_sys_state(self.sys._state) 
            self.ctrl_benchmarking.upd_accum_obj(observation, action)
            
            stage_obj = self.ctrl_benchmarking.stage_obj(observation, action)
            accum_obj = self.ctrl_benchmarking.accum_obj_val
        
        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        alpha_deg = alpha/np.pi*180
        
        cte = getattr(self.ros_node, "current_cte", 0.0) if hasattr(self, "ros_node") else 0.0
        heading_error = getattr(self.ros_node, "current_heading_error", 0.0) if hasattr(self, "ros_node") else 0.0
        steering = getattr(self.ros_node, "current_steering", 0.0) if hasattr(self, "ros_node") else 0.0
        
        cte = 0.0 if cte is None else cte
        heading_error = 0.0 if heading_error is None else heading_error
        steering = 0.0 if steering is None else steering

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, xCoord, yCoord, alpha, stage_obj, accum_obj, action)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, xCoord, yCoord, alpha, stage_obj, accum_obj, action, cte, heading_error, steering)
        
        # xy plane  
        text_time = 't = {time:2.3f}'.format(time = t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot
            
        self.robot_marker.rotate(1e-3)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(5, 5, marker=self.robot_marker.marker, s=400, c='b')       
        
        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord, yCoord, marker=self.robot_marker.marker, s=400, c='b')
        
        # # Solution
        #------------updated for qcar-----------------------------
        cte = 0.0
        if hasattr(self, "ros_node") and self.ros_node.current_cte is not None:
            cte = self.ros_node.current_cte

        upd_line(self.line_cte, t, cte)
        
        upd_line(self.line_stage_obj, t, stage_obj)
        upd_line(self.line_accum_obj, t, accum_obj)
	
        #------velocity--------------
        #speed_ref = getattr(self, "speed_ref", 0.0)
        #upd_line(self.line_velocity, t, speed_ref)
        
        velocity_measured = (getattr(self.ros_node, "current_velocity_measured", 0.0)
        if hasattr(self, "ros_node")
            else 0.0
        )

        upd_line(self.line_velocity, t, velocity_measured)
    
        # Cost
        heading_error = 0.0

        if hasattr(self, "ros_node") and \
            hasattr(self.ros_node, "current_heading_error") and \
            self.ros_node.current_heading_error is not None:

            heading_error = self.ros_node.current_heading_error

        upd_line(self.line_heading_error, t, heading_error)
        
        #text_accum_obj = r'$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}'.format(accum_obj = accum_obj)
        #upd_text(self.text_accum_obj_handle, text_accum_obj)
        
        # Control
        steering = 0.0

        if hasattr(self, "ros_node") and \
            hasattr(self.ros_node, "current_steering") and \
            self.ros_node.current_steering is not None:

            steering = self.ros_node.current_steering

        upd_line(self.line_steering, t, steering)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))  
            
            self.run_curr += 1
                    
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            accum_obj = 0  
               
            reset_line(self.line_stage_obj)
            reset_line(self.line_accum_obj)
            
            reset_line(self.line_cte)
            reset_line(self.line_heading_error)
            reset_line(self.line_steering)
            reset_line(self.line_velocity)
        
            upd_line(self.line_traj, np.nan, np.nan)
            
class Animator2Tank(Animator):
    """
    Animator class for a 2-tank system. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        state_init, \
        action_init, \
        t0, \
        t1, \
        state_full_init, \
        ctrl_mode, \
        action_manual, \
        action_min, \
        action_max, \
        Nruns, \
        is_print_sim_step, \
        is_log_data, \
        is_playback, \
        stage_obj_init, \
        level_target = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.state_full_init = state_full_init
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.action_manual = action_manual
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        self.level_target = level_target
        
        h1_0 = state_init[0]
        h2_0 = state_init[1]
        p0 = action_init
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))    
            
        # h1, h2 plot
        self.axs_sol = self.fig_sim.add_subplot(221, autoscale_on=False, xlim=(t0,t1), ylim=( -2, 2 ), xlabel='t [s]', title='Pause - space, q - quit, click - data cursor')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_sol.plot([t0, t1], [level_target[0], level_target[0]], 'b--', lw=0.75)   # Help line (target)
        self.axs_sol.plot([t0, t1], [level_target[1], level_target[1]], 'r--', lw=0.75)   # Help line (target)
        self.line_h1, = self.axs_sol.plot(t0, h1_0, 'b-', lw=0.5, label=r'$h_1$')
        self.line_h2, = self.axs_sol.plot(t0, h2_0, 'r-', lw=0.5, label=r'$h_2$') 
        self.axs_sol.legend(fancybox=True, loc='upper right')
        self.axs_sol.format_coord = lambda state,observation: '%2.2f, %2.2f' % (state,observation)
        
        # Cost
        if is_playback:
            stage_obj = stage_obj_init
        else:
            observation_init = self.sys.out(state_init)
            stage_obj = self.ctrl_benchmarking.stage_obj(observation_init, action_init)
        
        self.axs_cost = self.fig_sim.add_subplot(223, autoscale_on=False, xlim=(t0,t1), ylim=(0, 1e4*stage_obj), yscale='symlog', xlabel='t [s]')
        
        text_accum_obj = r'$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.3f}'.format(accum_obj = 0)
        self.text_accum_obj_handle = self.fig_sim.text(0.05, 0.5, text_accum_obj, horizontalalignment='left', verticalalignment='center')
        self.line_stage_obj, = self.axs_cost.plot(t0, stage_obj, 'r-', lw=0.5, label='Stage obj.')
        self.line_accum_obj, = self.axs_cost.plot(t0, 0, 'g-', lw=0.5, label=r'$\int \mathrm{Stage\,obj.} \,\mathrm{d}t$')
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Control
        self.axs_ctrl = self.fig_sim.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=(action_min-0.1, action_max+0.1), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_ctrl, = self.axs_ctrl.plot(t0, p0, lw=0.5, label='p') 
        self.axs_ctrl.legend(fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['line_h1', 'line_h2', 'line_stage_obj', 'line_accum_obj', 'line_ctrl'])
        self.lines = cLines(line_h1=self.line_h1,
                            line_h2=self.line_h2,
                            line_stage_obj=self.line_stage_obj,
                            line_accum_obj=self.line_accum_obj,
                            line_ctrl=self.line_ctrl)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)         
                
    def set_sim_data(self, ts, h1s, h2s, ps, rs, accum_objs):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.h1s, self.h2s, self.ps = ts, h1s, h2s, ps
        self.rs, self.accum_objs = rs, accum_objs
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.state_full = np.array([self.h1s[self.curr_step], self.h2s[self.curr_step]])
        self.stage_obj = self.rs[self.curr_step]
        self.accum_obj = self.accum_objs[self.curr_step]
        self.action = np.array([self.ps[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        state_init, *_ = self.pars      
        
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            state_full = self.state_full
            action = self.action
            stage_obj = self.stage_obj
            accum_obj = self.accum_obj        
            
        else:
            self.simulator.sim_step()
            
            t, state, observation, state_full = self.simulator.get_sim_step_data()
            
            action = self.ctrl_selector(t, observation, self.action_manual, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(action)
            self.ctrl_benchmarking.receive_sys_state(self.sys._state) 
            self.ctrl_benchmarking.upd_accum_obj(observation, action)
            
            stage_obj = self.ctrl_benchmarking.stage_obj(observation, action)
            accum_obj = self.ctrl_benchmarking.accum_obj_val
        
        h1 = state_full[0]
        h2 = state_full[1]
        p = action

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, h1, h2, p, stage_obj, accum_obj)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, h1, h2, p, stage_obj, accum_obj)
        
        # # Solution
        upd_line(self.line_h1, t, h1)
        upd_line(self.line_h2, t, h2)
    
        # Cost
        upd_line(self.line_stage_obj, t, stage_obj)
        upd_line(self.line_accum_obj, t, accum_obj)
        text_accum_obj = r'$\int \mathrm{{Stage\,obj.}} \,\mathrm{{d}}t$ = {accum_obj:2.1f}'.format(accum_obj = accum_obj)
        upd_text(self.text_accum_obj_handle, text_accum_obj)
        
        # Control
        upd_line(self.line_ctrl, t, p)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))
                
            self.run_curr += 1
            
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            accum_obj = 0     
            
            reset_line(self.line_h1)
            reset_line(self.line_h1)
            reset_line(self.line_ctrl)
            reset_line(self.line_stage_obj)
            reset_line(self.line_accum_obj)
            
            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)
    
            # upd_line(self.line_h1, np.nan)                


# QCar-specific visualizer: read-only playback, no calls to compute_action/sim_step.
from pathlib import Path
from threading import RLock
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.widgets import Button

class AnimatorQCar(Animator):
    """Record one sample per controller cycle, then plot or replay the results.

    The coordinate frame is rear axle [x, y, theta], steering is delta in rad.
    Cross-track error, heading error, stage cost and integral cost are read
    from the controller *after* its compute_action call has completed.
    """

    def __init__(self, path, controller_name='QCar', path_name='',
                 target_speed=None, wheelbase=0.256,
                 car_display_mode='physical'):
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[1] < 2:
            raise ValueError('Expected a reference path of shape (N, 2+)')
        self.controller_name = str(controller_name)
        self.path_name = str(path_name)
        self.target_speed = target_speed
        self.wheelbase = float(wheelbase)
        # physical: the car is a metric-scale polygon (same actual QCar size
        # on every trajectory). icon: keep its size proportional to the visible
        # plot span so it is legible even when the path radius changes.
        # Icon mode is VISUAL ONLY and is explicitly labeled not to scale.
        if car_display_mode not in ('physical', 'icon'):
            raise ValueError("car_display_mode must be 'physical' or 'icon'")
        self.car_display_mode = car_display_mode
        self.samples = []
        # ROS callbacks may record while the GUI reads. Plotting stays on the GUI thread.
        self._live_lock = RLock()
        self._external_path_received = False
        self.live_max_samples = None

    def update_live_path(self, path):
        """ROS transport calls this after accepting the external path. No ROS imports."""
        new_path = np.asarray(path, dtype=float)
        if new_path.ndim != 2 or new_path.shape[0] < 3 or new_path.shape[1] < 2:
            raise ValueError("A live path must contain at least three [x,y] points")
        if not np.all(np.isfinite(new_path)):
            raise ValueError("Nonfinite live path")
        with self._live_lock:
            self.path = new_path.copy()
            self._external_path_received = True

    def record(self, t, observation, action, controller):
        """Copy recorded sample; never call or modify the controller."""
        state = np.asarray(observation, dtype=float)
        cmd = np.asarray(action, dtype=float)
        row = [
            float(t), float(state[0]), float(state[1]), float(state[2]),
            float(cmd[0]), float(cmd[1]),
            float(controller.current_cte),
            float(controller.current_heading_error),
            float(controller.current_stage_obj),
            float(controller.accum_obj_val),
            float(getattr(controller, 'speed_reference_curr', self.target_speed or cmd[0])),
        ]
        with self._live_lock:
            self.samples.append(row)
            if self.live_max_samples and len(self.samples) > self.live_max_samples:
                # Only the screen buffer is bounded; the CSV logger retains the full run.
                del self.samples[:len(self.samples) - self.live_max_samples]

    @staticmethod
    def _limits(values, include_zero=True, min_span=0.05):
        values = np.asarray(values, dtype=float)
        lo = float(np.min(values))
        hi = float(np.max(values))
        if include_zero:
            lo = min(lo, 0.)
            hi = max(hi, 0.)
        width = max(hi-lo, min_span)
        mid = (lo+hi)/2
        return mid-width*.58, mid+width*.58

    def _data(self):
        with self._live_lock:
            if not self.samples:
                raise RuntimeError('No simulation samples to display')
            data = np.asarray(self.samples, dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError('Nonfinite samples cannot be animated')
        return data

    def _make_figure(self, data):
        """Construct a persistent Matplotlib dashboard; update its artists only."""
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
        fig.subplots_adjust(left=.07, right=.96, bottom=.13, top=.87,
                            wspace=.29, hspace=.36)
        ax_path, ax_cte, ax_cost = axs[0]
        ax_heading, ax_steer, ax_speed = axs[1]
        for ax in axs.ravel():
            ax.grid(True, alpha=.25)
            ax.tick_params(labelsize=9)
        t = data[:, 0]
        x, y, theta = data[:, 1], data[:, 2], data[:, 3]

        # World view includes full reference trajectory and realized trajectory.
        ref_line, = ax_path.plot(self.path[:, 0], self.path[:, 1], '--', lw=1.4,
                                label='Reference', color='tab:blue')
        trace, = ax_path.plot([], [], '-', lw=1.9, label='QCar trajectory',
                              color='tab:orange')
        dot, = ax_path.plot([], [], 'o', color='tab:red', ms=3)
        start_dot, = ax_path.plot(x[0], y[0], 'go', ms=5, label='Start')
        body = Polygon(np.zeros((4, 2)), closed=True, fc='royalblue',
                       ec='navy', lw=1.2, zorder=6)
        ax_path.add_patch(body)
        world = np.vstack((self.path[:, :2], data[:, 1:3]))
        xmin, ymin = world.min(axis=0)
        xmax, ymax = world.max(axis=0)
        pad = max(float(np.max([xmax-xmin, ymax-ymin]))*.1, .2)
        ax_path.set_xlim(xmin-pad, xmax+pad)
        ax_path.set_ylim(ymin-pad, ymax+pad)
        ax_path.set_aspect('equal', adjustable='box')
        ax_path.set_xlabel('x [m]')
        ax_path.set_ylabel('y [m]')
        if self.car_display_mode == 'icon':
            ax_path.set_title('QCar trajectory (car icon not to scale)')
        else:
            ax_path.set_title('QCar trajectory')
        ax_path.legend(loc='best', fontsize=8)

        def time_axis(ax, values, label, color, title, min_span):
            ax.set_xlim(float(t[0]), max(float(t[-1]), float(t[0])+.1))
            ax.set_ylim(*self._limits(values, min_span=min_span))
            line, = ax.plot([], [], color=color, lw=1.45, label=label)
            ax.axhline(0, color='0.35', ls=':', lw=.8)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(label)
            ax.set_title(title)
            return line

        cte_line = time_axis(ax_cte, data[:, 6], 'CTE [m]', 'tab:blue',
                             'Cross-track error', min_span=.03)
        heading_line = time_axis(ax_heading, data[:, 7], 'Heading error [rad]',
                                 'tab:purple', 'Heading error', min_span=.04)
        steer_line = time_axis(ax_steer, data[:, 5], 'Steering [rad]',
                               'tab:orange', 'Steering angle', min_span=.04)
        velocity_range = np.r_[data[:, 4], data[:, 10]]
        if self.target_speed is not None:
            velocity_range = np.r_[velocity_range, float(self.target_speed)]
        speed_line = time_axis(ax_speed, velocity_range, 'Velocity [m/s]',
                               'tab:green', 'Forward velocity', min_span=.1)
        speed_ref_line, = ax_speed.plot([], [], color='tab:blue', ls='--',
                                       lw=1.4, label='Curve speed reference')
        if self.target_speed is not None:
            ax_speed.axhline(float(self.target_speed), color='0.3',
                            ls=':', lw=1, label='Cruise target')
        ax_speed.legend(loc='best', fontsize=8)

        # Cost terms differ in scale: use TWO y axes, not one log scale that
        # visually erases the instantaneous cost when accumulated cost grows.
        cost_line = time_axis(ax_cost, data[:, 8], 'Stage cost', 'tab:red',
                              'Tracking objective', min_span=.1)
        ax_cost_right = ax_cost.twinx()
        ax_cost_right.set_ylim(*self._limits(data[:, 9], min_span=.1))
        ax_cost_right.set_ylabel('Accumulated cost', fontsize=9)
        acc_line, = ax_cost_right.plot([], [], color='tab:green', lw=1.5,
                                      label='Accumulated cost')
        ax_cost.legend(handles=[cost_line, acc_line], loc='best', fontsize=8)

        status = fig.suptitle('', fontsize=13)
        info = fig.text(.5, .032,
                        'Space: pause / resume   |   ← / →: move in time   |   R: restart   |   Q: quit',
                        ha='center', fontsize=9)
        lines = [cte_line, cost_line, acc_line, heading_line,
                 steer_line, speed_line, speed_ref_line]
        artists = dict(trace=trace, dot=dot, start_dot=start_dot, body=body, lines=lines,
                       reference_line=ref_line, cost_right_axis=ax_cost_right,
                       chart_axes=(ax_cte, ax_cost, ax_heading, ax_steer, ax_speed),
                       status=status, info=info, path_axis=ax_path)
        return fig, artists

    def _paint(self, data, artists, idx):
        """Set all artists to the recorded values up to idx, no simulation."""
        idx = int(np.clip(idx, 0, len(data)-1))
        rows = data[:idx+1]
        x, y, yaw = data[idx, 1:4]
        artists['trace'].set_data(rows[:, 1], rows[:, 2])
        artists['dot'].set_data([x], [y])
        # QCar polygon is anchored to the rear axle (the observation origin).
        # By default its geometry is drawn in metres, so a larger trajectory
        # *correctly* makes the car appear smaller in the world-view panel.
        # Optional 'icon' mode maintains a legible size for presentations.
        physical_length = max(.32, self.wheelbase*1.35)
        if self.car_display_mode == 'physical':
            length = physical_length
        else:
            ax = artists['path_axis']
            xspan = abs(float(np.diff(ax.get_xlim())[0]))
            yspan = abs(float(np.diff(ax.get_ylim())[0]))
            # Constant share of the world-view extent, not the actual metres.
            length = .13 * max(xspan, yspan)
        width = length * (.17 / physical_length)
        rear_offset = length * (.08 / physical_length)
        local = np.array([[-rear_offset, -width/2],
                          [length-rear_offset, -width/2],
                          [length-rear_offset, width/2],
                          [-rear_offset, width/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        artists['body'].set_xy(local@rot.T + np.array([x, y]))
        time = rows[:, 0]
        for line, values in zip(artists['lines'],
                (rows[:, 6], rows[:, 8], rows[:, 9], rows[:, 7],
                 rows[:, 5], rows[:, 4], rows[:, 10])):
            line.set_data(time, values)
        artists['status'].set_text(
            f'{self.controller_name} — {self.path_name}  |  '
            f't = {data[idx, 0]:.2f} s  |  '
            f'CTE = {data[idx, 6]:+.4f} m  |  '
            f'v = {data[idx, 4]:.3f} m/s')

    def show_live(self, refresh_hz=5.0):
        """View ROS/Gazebo samples live, WITHOUT invoking a controller or simulator.

        Call from the *main GUI thread* while a single ROS executor spins in a
        background thread. ``record`` and ``update_live_path`` are called only
        by ROS callbacks. Uses Matplotlib's GUI timer to read snapshots.
        Closing the window returns so the caller can safely shut down ROS.
        """
        refresh_hz = float(refresh_hz)
        if not 0.5 <= refresh_hz <= 30.:
            raise ValueError('refresh_hz must be between 0.5 and 30')
        backend = str(plt.get_backend()).lower()
        if backend in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template') or 'inline' in backend:
            raise RuntimeError(f'Matplotlib backend {backend!r} cannot display a live window. '
                               'Install python3-tk and choose TkAgg or a working Qt backend.')

        # One synthetic row is used solely to initialize blank axes; it is NEVER
        # displayed or logged as a real observation or controller sample.
        blank = np.zeros((1, 11), dtype=float)
        fig, artists = self._make_figure(blank)
        artists['reference_line'].set_data([], [])
        artists['start_dot'].set_data([], [])  # No false 'start' at (0,0) while waiting
        artists['info'].set_text('Gazebo odometry → rcognita commands  |  Close window to stop')
        artists['status'].set_text('Waiting for external path, odometry and controller samples…')
        self.live_max_samples = 15000
        previous = {'path_version': None}

        def update_dashboard():
            if not plt.fignum_exists(fig.number):
                return
            with self._live_lock:
                received_path = self._external_path_received
                live_path = self.path.copy() if received_path else None
                data = np.asarray(self.samples, dtype=float).copy() if self.samples else None
            if live_path is not None:
                artists['reference_line'].set_data(live_path[:, 0], live_path[:, 1])
            if data is None or not len(data):
                artists['status'].set_text('Waiting for valid path + odometry + control timer…')
                fig.canvas.draw_idle()
                return
            # Limit GUI draw work; the CSV still retains every controller cycle.
            plotted = data[::max(1, len(data)//2000)]
            if plotted[-1, 0] != data[-1, 0]:
                plotted = np.vstack((plotted, data[-1]))
            self._paint(plotted, artists, len(plotted)-1)
            artists['start_dot'].set_data([data[0, 1]], [data[0, 2]])
            last_t = float(data[-1, 0])
            axes = artists['chart_axes']
            for ax in axes:
                ax.set_xlim(0., max(60., last_t + 1.))
            metric_cols = (6, 8, 7, 5, 4)
            min_spans = (.03, .1, .04, .04, .1)
            for ax, col, span in zip(axes, metric_cols, min_spans):
                measure = data[:, col]
                if col == 4:
                    measure = np.r_[measure, data[:, 10]]
                    if self.target_speed is not None:
                        measure = np.r_[measure, self.target_speed]
                ax.set_ylim(*self._limits(measure, min_span=span))
            artists['cost_right_axis'].set_ylim(*self._limits(data[:, 9], min_span=.1))
            locations = [data[:, 1:3]]
            if live_path is not None:
                locations.insert(0, live_path[:, :2])
            xy = np.vstack(locations)
            low = xy.min(axis=0)
            high = xy.max(axis=0)
            padding = max(.2, float(np.max(high-low))*.12)
            artists['path_axis'].set_xlim(low[0]-padding, high[0]+padding)
            artists['path_axis'].set_ylim(low[1]-padding, high[1]+padding)
            fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=int(1000./refresh_hz))
        timer.add_callback(update_dashboard)
        update_dashboard()
        timer.start()
        try:
            plt.show(block=True)
        finally:
            timer.stop()
            plt.close(fig)

    def save_dashboard(self, output_path):
        """Save the 6-panel figure at the END of the simulation."""
        data = self._data()
        fig, artists = self._make_figure(data)
        self._paint(data, artists, len(data)-1)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=140)
        plt.close(fig)
        return output

    def play(self, show=False, output_path=None, fps=20, frame_stride=1):
        """Replay the recorded run with a moving car and six updating panels.

        show=True opens a Matplotlib GUI; output_path saves .gif (Pillow) or
        .mp4 (ffmpeg needed). Neither function invokes a controller or simulator.
        """
        data = self._data()
        fps = int(fps)
        frame_stride = int(frame_stride)
        if fps <= 0 or frame_stride <= 0:
            raise ValueError('fps and frame_stride must be positive integers')
        if not show and not output_path:
            raise ValueError('Provide show=True and/or an output_path')
        frames = list(range(0, len(data), frame_stride))
        if frames[-1] != len(data)-1:
            frames.append(len(data)-1)
        fig, artists = self._make_figure(data)

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            ext = output.suffix.lower()
            if ext == '.gif':
                writer = animation.PillowWriter(fps=fps)
            elif ext == '.mp4':
                writer = animation.FFMpegWriter(fps=fps)
            else:
                plt.close(fig)
                raise ValueError('Animation output must end in .gif or .mp4')

            def export_step(k):
                self._paint(data, artists, frames[k])
                return []

            export_anim = animation.FuncAnimation(
                fig, export_step, frames=len(frames), interval=1000/fps,
                repeat=False, blit=False, cache_frame_data=False)
            export_anim.save(str(output), writer=writer, dpi=85)

        if show:
            # This callback displays precomputed samples, NEVER computes control.
            cursor = {'index': 0, 'paused': False}
            ax_button = fig.add_axes((.82, .025, .115, .045))
            button = Button(ax_button, 'Pause', hovercolor='0.9')
            self._paint(data, artists, frames[0])

            def toggle(_=None):
                cursor['paused'] = not cursor['paused']
                if cursor['paused']:
                    replay_anim.event_source.stop()
                    button.label.set_text('Replay' if cursor['index'] == len(frames)-1
                                          else 'Resume')
                else:
                    if cursor['index'] == len(frames)-1:
                        cursor['index'] = 0
                    replay_anim.event_source.start()
                    button.label.set_text('Pause')
                fig.canvas.draw_idle()

            def seek(delta):
                cursor['index'] = int(np.clip(cursor['index'] + delta, 0,
                                              len(frames)-1))
                self._paint(data, artists, frames[cursor['index']])
                fig.canvas.draw_idle()

            def on_key(event):
                if event.key in (' ', 'p'):
                    toggle()
                elif event.key == 'left':
                    if not cursor['paused']:
                        toggle()
                    seek(-max(1, fps//2))
                elif event.key == 'right':
                    if not cursor['paused']:
                        toggle()
                    seek(max(1, fps//2))
                elif event.key == 'r':
                    cursor['index'] = 0
                    self._paint(data, artists, frames[0])
                    if cursor['paused']:
                        toggle()
                elif event.key == 'q':
                    plt.close(fig)

            def playback_step(_):
                if cursor['paused']:
                    return []
                self._paint(data, artists, frames[cursor['index']])
                if cursor['index'] < len(frames)-1:
                    cursor['index'] += 1
                else:
                    cursor['paused'] = True
                    button.label.set_text('Replay')
                    replay_anim.event_source.stop()
                return []

            replay_anim = animation.FuncAnimation(
                fig, playback_step, frames=None,
                interval=1000/fps, repeat=True, blit=False,
                cache_frame_data=False)
            button.on_clicked(toggle)
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()
        plt.close(fig)
        return output_path

    @property
    def history(self):
        """Recorded xy points; kept for the preset summary."""
        return [np.asarray(row[1:3]) for row in self.samples]

    def render(self, output_path, show=False):
        """Draw a trajectory-only PNG from the same recorded state."""
        data = self._data()
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(self.path[:, 0], self.path[:, 1], 'b--', label='Reference')
        ax.plot(data[:, 1], data[:, 2], color='tab:orange', label='QCar trajectory')
        ax.plot(data[0, 1], data[0, 2], 'go', label='Start')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title('QCar path tracking'); ax.grid(True, alpha=.3); ax.legend()
        fig.savefig(output_path, dpi=120, bbox_inches='tight')
        if show: plt.show()
        plt.close(fig)
