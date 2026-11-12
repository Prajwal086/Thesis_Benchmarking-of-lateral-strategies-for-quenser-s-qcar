#!/usr/bin/env python3
"""Standalone QCar rcognita preset: ONLY rcognita, numpy, scipy; NO ROS imports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from rcognita import simulator, loggers, visuals
from rcognita.systems import SysQCar
from rcognita.controllers import CtrlPID, CtrlPurePursuit, CtrlStanley, CtrlMPC
from rcognita.qcar_paths import make_path


def add_arguments(parser):
    """Register shared path, vehicle, and controller tuning options.

    Controller-specific options are used only when their controller is selected.
    Each option's help text describes its effect and units where applicable.
    """
    parser.add_argument('--ctrl_mode', choices=['PP','PID','Stanley','MPC'], default='Stanley',
                        help='Steering controller: Pure Pursuit, PID, Stanley, or model-predictive control')
    parser.add_argument('--path_type', choices=['circle','figure_eight','infinity','straight',
                                               's_curve','mild_curve'], default='circle',
                        help='Shape of the generated reference path')
    parser.add_argument('--path_radius', type=float, default=2.5,
                        help='Radius in metres for circular portions of the generated path')
    parser.add_argument('--path_points', type=int, default=240,
                        help='Number of waypoints used to sample the generated path')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Initial path waypoint; on a figure-eight, choose a point away from the crossing')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Controller and simulation sampling period in seconds')
    parser.add_argument('--wheelbase', type=float, default=0.25725,
                        help='Rear-to-front axle distance in metres used by the vehicle model')
    parser.add_argument('--target_speed', type=float, default=1.0,
                        help='Desired forward speed in metres per second, clipped to the speed limits')
    parser.add_argument('--speed_policy', choices=['constant','adaptive'], default='adaptive',
                        help='Constant holds target speed; adaptive lowers the reference speed on tighter curves')
    parser.add_argument('--min_speed', type=float, default=0.08,
                        help='Lower bound for the forward-speed reference in metres per second')
    parser.add_argument('--max_speed', type=float, default=1.00,
                        help='Upper bound for the forward-speed reference in metres per second')
    parser.add_argument('--max_steering', type=float, default=0.5236,
                        help='Magnitude limit for front-wheel steering in radians')
    parser.add_argument('--max_lateral_acceleration', type=float, default=0.12,
                        help='Lateral-acceleration limit in m/s^2 used to reduce speed on curves')
    parser.add_argument('--curve_speed_safety_factor', type=float, default=0.92,
                        help='Multiplier on the curvature-based speed limit; lower values are more conservative')
    parser.add_argument('--curvature_deadband', type=float, default=0.02,
                        help='Treat path curvature magnitudes below this 1/m threshold as straight')
    parser.add_argument('--closed_path_gap_factor', type=float, default=2.5,
                        help='Endpoint-gap-to-median-spacing threshold for detecting a closed path')
    parser.add_argument('--max_accel', type=float, default=1.2,
                        help='Maximum reference-speed increase in m/s^2')
    parser.add_argument('--max_decel', type=float, default=1.2,
                        help='Maximum reference-speed decrease in m/s^2')
    parser.add_argument('--lookahead_distance', type=float, default=0.5,
                        help='Pure Pursuit target distance ahead of the rear axle, in metres')
    parser.add_argument('--stanley_k', type=float, default=0.8,
                        help='Stanley cross-track-error gain; larger values increase correction for lateral error')
    parser.add_argument('--softening_gain', type=float, default=0.1,
                        help='Stanley denominator offset added to speed to soften low-speed steering')
    parser.add_argument('--min_velocity', type=float, default=0.1,
                        help='Minimum absolute speed used in the Stanley steering law, in m/s')
    parser.add_argument('--kp', type=float, default=0.85,
                        help='PID proportional gain on cross-track error')
    parser.add_argument('--ki', type=float, default=0.005,
                        help='PID integral gain on accumulated cross-track error')
    parser.add_argument('--kd', type=float, default=0.32,
                        help='PID derivative gain on cross-track error')
    parser.add_argument('--heading_gain', type=float, default=0.6,
                        help='PID steering contribution from path-heading error')
    parser.add_argument('--curvature_feedforward_gain', type=float, default=1.0,
                        help='PID multiplier on curvature-based steering feedforward')
    parser.add_argument('--Nactor', type=int, default=15,
                        help='MPC number of control/prediction steps in its receding horizon')
    parser.add_argument('--pred_step_size_multiplier', type=float, default=2.0,
                        help='MPC prediction-step duration as a multiple of --dt')
    parser.add_argument('--optimizer_max_iterations', type=int, default=25,
                        help='Maximum optimizer iterations per MPC update')
    parser.add_argument('--optimizer_ftol', type=float, default=1.e-5,
                        help='MPC optimizer objective-change tolerance for convergence')
    parser.add_argument('--w_steering_rate_violation', type=float, default=200.,
                        help='MPC penalty weight for steering-rate-limit violations')
    parser.add_argument('--w_speed_overshoot', type=float, default=250.,
                        help='MPC penalty weight for speed above the curvature-based speed envelope')
    parser.add_argument('--mpc_max_acceleration', type=float, default=0.8,
                        help='MPC maximum commanded acceleration in m/s^2')
    parser.add_argument('--mpc_max_deceleration', type=float, default=1.2,
                        help='MPC maximum commanded deceleration magnitude in m/s^2')
    parser.add_argument('--max_steering_rate', type=float, default=2.0,
                        help='MPC maximum steering change rate in radians per second')
    parser.add_argument('--w_cte', type=float, default=20.0,
                        help='Stage-cost weight on squared cross-track error')
    parser.add_argument('--w_heading', type=float, default=10.0,
                        help='Stage-cost weight on squared path-heading error')
    parser.add_argument('--w_speed', type=float, default=35.0,
                        help='Stage-cost weight on squared difference from the speed reference')
    parser.add_argument('--w_command_speed', type=float, default=0.0,
                        help='Stage-cost weight on squared commanded/predicted speed magnitude')
    parser.add_argument('--w_steering', type=float, default=0.1,
                        help='Stage-cost weight on squared steering magnitude')
    return parser


def build(args):
    """Same classes and parameter dictionaries for plain and ROS 2 presets."""
    path = make_path(args.path_type, args.path_points, args.path_radius)
    system = SysQCar(args.wheelbase,args.min_speed,args.max_speed,args.max_steering)
    base = dict(path=path, wheelbase=args.wheelbase, sampling_time=args.dt,
                target_speed=args.target_speed, min_speed=args.min_speed,
                max_speed=args.max_speed, max_steering=args.max_steering,
                max_lateral_acceleration=args.max_lateral_acceleration,
                curve_speed_safety_factor=args.curve_speed_safety_factor,
                curvature_deadband=args.curvature_deadband,
                max_accel=args.max_accel, max_decel=args.max_decel,
                closed_path_gap_factor=args.closed_path_gap_factor,
                speed_policy=args.speed_policy,
                stage_obj_pars=dict(w_e=args.w_cte, w_heading=args.w_heading,
                                    w_speed=args.w_speed,
                                    w_command_speed=args.w_command_speed,
                                    w_steering=args.w_steering))
    if args.ctrl_mode == 'PP':
        extra = dict(lookahead_distance=args.lookahead_distance)
        clazz = CtrlPurePursuit
    elif args.ctrl_mode == 'PID':
        extra = dict(kp=args.kp,ki=args.ki,kd=args.kd,heading_gain=args.heading_gain,
                     curvature_feedforward_gain=args.curvature_feedforward_gain)
        clazz = CtrlPID
    elif args.ctrl_mode == 'Stanley':
        extra = dict(k=args.stanley_k,softening_gain=args.softening_gain,
                     min_velocity=args.min_velocity)
        clazz = CtrlStanley
    else:
        extra = dict(sys_rhs=system._state_dyn, Nactor=args.Nactor,
                     pred_step_size=args.dt*args.pred_step_size_multiplier,
                     optimizer_max_iterations=args.optimizer_max_iterations,
                     optimizer_ftol=args.optimizer_ftol,
                     w_steering_rate_violation=args.w_steering_rate_violation,
                     w_speed_overshoot=args.w_speed_overshoot,
                     max_acceleration=args.mpc_max_acceleration,
                     max_deceleration=args.mpc_max_deceleration,
                     max_steering_rate=args.max_steering_rate)
        clazz = CtrlMPC
    controller = clazz(**base, **extra)
    # Only the active controller's parameters are included in metadata.
    active = {k: v for k, v in {**base, **extra}.items() if k not in ('path','sys_rhs')}
    return system, controller, path, active


def run(args):
    system, controller, path, params = build(args)
    # The figure-eight path starts at its crossing. Starting 0.3 m off that
    # crossing can project onto the wrong branch, particularly for Stanley's
    # front-axle tracking point. Select a non-crossing index for that scenario.
    if not (0 <= args.start_index < len(path)):
        raise ValueError(f'--start_index must be in [0,{len(path)-1}]')
    start = path[args.start_index, :2].copy()
    forward = np.array([np.cos(path[args.start_index, 2]),
                        np.sin(path[args.start_index, 2])])
    start += args.start_offset*np.array([-forward[1],forward[0]])
    state_init = np.r_[start, path[args.start_index, 2]+args.heading_offset]

    # Geometric feasibility does not improve with reduced vehicle speed: a
    # curvature beyond tan(max_steering)/wheelbase needs more steering travel.
    path_max_curvature = float(np.max(np.abs(controller._seg_curvature)))
    car_max_curvature = float(np.tan(args.max_steering)/args.wheelbase)
    if path_max_curvature > car_max_curvature * (1.0 + 1e-6):
        print('WARNING: reference path curvature exceeds QCar steering geometry: '
              f'max |kappa_path|={path_max_curvature:.3f} 1/m, '
              f'max |kappa_QCar|={car_max_curvature:.3f} 1/m. '
              'Slowing down alone cannot eliminate the error; use a '
              'larger path radius/smoother trajectory or revise physical steering limits.')
    system.receive_action(np.array([args.min_speed, 0.]))
    # Use rcognita's discrete Simulator with fixed-step RK4 of EXACTLY the
    # shared SysQCar RHS. This keeps the simulation/controller on one time base;
    # a variable-step RK45 would otherwise overshoot sampling instants.
    def discrete_step(t, state):
        h = args.dt
        action = system.action.copy()
        dyn = system._state_dyn
        a = dyn(t, state, action, ())
        b = dyn(t+h/2, state+h*a/2, action, ())
        c = dyn(t+h/2, state+h*b/2, action, ())
        d = dyn(t+h, state+h*c, action, ())
        state_next = state+(h/6)*(a+2*b+2*c+d)
        state_next[2] = np.arctan2(np.sin(state_next[2]),np.cos(state_next[2]))
        return state_next
    my_simulator = simulator.Simulator(
        sys_type='discr_fnc',closed_loop_rhs=discrete_step,
        sys_out=system.out, state_init=state_init,
        action_init=np.array([args.min_speed,0.]),t0=0.,t1=args.t1,
        dt=args.dt)
    metadata=dict(system='SysQCar',controller=type(controller).__name__,
                  path_type=args.path_type,path_radius=args.path_radius,
                  dt=args.dt,t1=args.t1,start_offset=args.start_offset,
                  start_index=args.start_index, heading_offset=args.heading_offset,
                  path_max_curvature=path_max_curvature,
                  qcar_max_curvature=car_max_curvature, parameters=params)
    logger = loggers.LoggerQCar(args.csv,metadata)
    animator = visuals.AnimatorQCar(path, args.ctrl_mode, args.path_type,
                                   target_speed=args.target_speed, wheelbase=args.wheelbase)
    want_dashboard = bool(getattr(args, "animate", False) or getattr(args, "save_animation", "") or getattr(args, "dashboard_png", ""))
    dashboard = animator if want_dashboard else None
    for k in range(int(np.floor(args.t1/args.dt))):
        t = k*args.dt
        observation = np.asarray(my_simulator.observation)
        action = controller.compute_action(t, observation)
        system.receive_action(action.copy())
        logger.log(t,observation,action,controller)
        animator.record(t, observation, action, controller)
        my_simulator.sim_step()
    if args.plot:
        animator.render(args.plot,show=False)
    if dashboard is not None and getattr(args, "dashboard_png", ""):
        dashboard.save_dashboard(args.dashboard_png)
    if dashboard is not None and (getattr(args, "animate", False) or getattr(args, "save_animation", "")):
        show = getattr(args, "animate", False)
        if show:
            import matplotlib
            backend = str(matplotlib.get_backend()).lower()
            if backend in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template') or 'inline' in backend:
                print(f'WARNING: Matplotlib backend {backend!r} is not interactive. '
                      'Use a GUI backend (TkAgg/QtAgg) for --animate, or '
                      '--save_animation output.gif for headless output.')
                show = False
        if show or getattr(args, "save_animation", ""):
            dashboard.play(show=show, output_path=args.save_animation,
                           fps=args.fps, frame_stride=args.frame_stride)
    print(f'{args.ctrl_mode} {args.path_type}: {len(animator.history)} cycles; '
          f'last CTE={controller.current_cte:.3f} m; '
          f'accum_obj={controller.accum_obj_val:.2f}; CSV={args.csv}')
    return controller


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--t1',type=float,default=25.,
                        help='Total standalone simulation duration in seconds')
    parser.add_argument('--start_offset',type=float,default=0.3,
                        help='Initial lateral displacement from the selected path point in metres; positive is left')
    parser.add_argument('--heading_offset',type=float,default=0.,
                        help='Initial yaw offset from the path heading in radians')
    parser.add_argument('--csv',default='qcar_results.csv',
                        help='Output CSV file for controller and tracking results')
    parser.add_argument('--plot', default='', help='Optional legacy trajectory-only PNG')
    parser.add_argument('--animate', action='store_true',
                        help='Open animated 6-panel Matplotlib dashboard (GUI)')
    parser.add_argument('--save_animation', default='',
                        help='Export 6-panel dashboard animation to .gif or .mp4')
    parser.add_argument('--dashboard_png', default='',
                        help='Save final 6-panel dashboard as PNG')
    parser.add_argument('--fps', type=int, default=20,
                        help='Animation playback frames per second (default: 20)')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Show every Nth simulation sample (default: 1)')
    args=parser.parse_args()
    if args.t1<=0: parser.error('--t1 must be >0')
    if args.fps<=0: parser.error('--fps must be >0')
    if args.frame_stride<=0: parser.error('--frame_stride must be >0')
    run(args)


if __name__=='__main__':
    main()
