#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the logger interface along with concrete realizations for each separate system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from tabulate import tabulate

import csv
import os

class Logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.Logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required).
        | :func:`~loggers.Logger.log_data_row` :
        | same as above, but write to a file (required).
    
    """
    
    def print_sim_step():
        pass
    
    def log_data_row():
        pass
    
class Logger3WRobot(Logger):
    """
    Data logger for a 3-wheel robot with dynamic actuators.
    
    """
    def print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'stage_obj', 'accum_obj', 'F [N]', 'M [N m]']  
        row_data = [t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action[0], action[1]]  
        row_format = ('8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action[0], action[1]])

class Logger3WRobotNI(Logger):
    """Data logger for a 3-wheel robot with static actuators.

    The live visualization API is intentionally unchanged.  For file logging,
    this class writes the same controller-metrics schema used by the
    ackermann_control MetricsLogger and obtains performance data directly from
    controllers.py, so visuals.py does not need to be modified.
    """

    ACKERMANN_HEADER = [
        'time',
        'cte',
        'heading_error',
        'velocity',
        'steering_cmd',
        'controller_loop_frequency_hz',
        'controller_execution_time_ms',
        'controller_cpu_usage_percent',
        'controller_deadline_miss',
        'controller_deadline_miss_ratio',
        'controller_cycle_count',
        'controller_deadline_miss_count',
    ]

    def __init__(self):
        self._initialized_metric_files = set()
        self._last_logged_sample_by_file = {}

    def print_sim_step(self, t, xCoord, yCoord, alpha, stage_obj, accum_obj, action):
    # alphaDeg = alpha/np.pi*180

        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'stage_obj', 'accum_obj', 'v [m/s]', 'omega [rad/s]']
        row_data = [t, xCoord, yCoord, alpha, stage_obj, accum_obj, action[0], action[1]]
        row_format = ('8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')

        print(table)

    @staticmethod
    def _controller_performance_samples():
        """Return every controller-cycle sample accumulated since the last log call."""
        try:
            from .controllers import get_controller_performance_samples
        except (ImportError, ValueError):
            try:
                from controllers import get_controller_performance_samples
            except ImportError:
                return []
        try:
            return get_controller_performance_samples()
        except Exception:
            # Logging must never interfere with the controller/simulation.
            return []

    def _ensure_ackermann_header(self, datafile):
        """Ensure a clean Ackermann-style CSV, removing rcognita metadata once."""
        datafile_key = os.path.abspath(str(datafile))
        if datafile_key in self._initialized_metric_files:
            return datafile_key

        already_ready = False
        try:
            with open(datafile, 'r', newline='') as infile:
                first_row = next(csv.reader(infile), None)
                already_ready = first_row == self.ACKERMANN_HEADER
        except (OSError, StopIteration):
            already_ready = False

        if not already_ready:
            with open(datafile, 'w', newline='') as outfile:
                csv.writer(outfile).writerow(self.ACKERMANN_HEADER)

        self._initialized_metric_files.add(datafile_key)
        return datafile_key

    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, stage_obj,
                     accum_obj, action, cte=0.0, heading_error=0.0,
                     steering=0.0):
        """Drain and write all controller-cycle rows accumulated since last call.

        The previous implementation kept only one global "latest snapshot".  If
        visuals.py/logging ran at 15-20 Hz while the controller ran faster, several
        controller cycles were overwritten before the logger saw them.  Draining a
        queue preserves one CSV row per actual controller cycle.
        """
        self._ensure_ackermann_header(datafile)
        samples = self._controller_performance_samples()

        # Instrumented controllers intentionally write only completed controller
        # cycles.  Do not create repeated/NaN rows when the visualization logger
        # happens to run between controller cycles.
        if not samples:
            return

        with open(datafile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            for telemetry in samples:
                if not telemetry.get('valid', False):
                    continue
                writer.writerow([
                    telemetry.get('sim_time', float('nan')),
                    telemetry.get('cte', cte),
                    telemetry.get('heading_error', heading_error),
                    telemetry.get('velocity', action[0]),
                    telemetry.get('steering_cmd', steering),
                    telemetry.get('controller_loop_frequency_hz', float('nan')),
                    telemetry.get('controller_execution_time_ms', float('nan')),
                    telemetry.get('controller_cpu_usage_percent', float('nan')),
                    telemetry.get('controller_deadline_miss', float('nan')),
                    telemetry.get('controller_deadline_miss_ratio', float('nan')),
                    int(telemetry.get('controller_cycle_count', 0)),
                    int(telemetry.get('controller_deadline_miss_count', 0)),
                ])

class Logger2Tank(Logger):
    """
    Data logger for a 2-tank system.
    
    """
    def print_sim_step(self, t, h1, h2, p, stage_obj, accum_obj):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'h1', 'h2', 'p', 'stage_obj', 'accum_obj']  
        row_data = [t, h1, h2, p, stage_obj, accum_obj]  
        row_format = ('8.1f', '8.4f', '8.4f', '8.4f', '8.4f', '8.2f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, h1, h2, p, stage_obj, accum_obj):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, h1, h2, p, stage_obj, accum_obj])                


class LoggerQCar(Logger):
    """Independent QCar logger; does not access any 3WRobotNI telemetry.

    Required columns are first. Extra columns separate the controller's own
    tracking error from the common rear-axle benchmark tracking error and
    capture per-instance control timing. Controller settings are written to
    a neighboring .meta.json file.
    """
    HEADER = [
        't', 'x', 'y', 'theta', 'v', 'delta', 'cte', 'heading_error',
        'stage_obj', 'accum_obj', 'benchmark_cte', 'benchmark_heading_error',
        'v_measured', 'controller_loop_frequency_hz',
        'controller_execution_time_ms', 'controller_cpu_usage_percent',
        'controller_deadline_miss', 'controller_deadline_miss_ratio',
        'controller_cycle_count', 'controller_deadline_miss_count',
        'speed_reference',
    ]

    def __init__(self, csv_path, metadata):
        import json
        from pathlib import Path
        self.realtime_environment = metadata.get('environment') == 'ROS2_Gazebo'
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('w', newline='') as f:
            csv.writer(f).writerow(self.HEADER)
        with self.path.with_suffix('.meta.json').open('w') as f:
            json.dump(metadata, f, indent=2)

    def log(self, t, observation, action, controller):
        import numpy as np
        obs = np.asarray(observation, dtype=float)
        command = np.asarray(action, dtype=float)
        benchmark = controller._geometry(obs, tracking_axle='rear')
        performance = dict(getattr(controller, 'performance', {}))
        # In unpaced offline simulation, callbacks execute much faster than
        # real time. Wall-loop rate and CPU load must not be presented as
        # measured 20 Hz ROS execution performance. Still record per-call
        # execution time for optimization diagnostics.
        if not self.realtime_environment:
            performance['controller_loop_frequency_hz'] = float('nan')
            performance['controller_cpu_usage_percent'] = float('nan')
        with self.path.open('a', newline='') as f:
            csv.writer(f).writerow([
                float(t), *map(float, obs[:3]),
                float(command[0]), float(command[1]),
                float(controller.current_cte),
                float(controller.current_heading_error),
                float(controller.current_stage_obj),
                float(controller.accum_obj_val),
                float(benchmark['cte']), float(benchmark['heading_error']),
                float(obs[3]) if obs.size > 3 else float('nan'),
                *(performance.get(field, float('nan')) for field in self.HEADER[13:20]),
                float(controller.speed_reference_curr),
            ])
