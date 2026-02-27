import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


class LateralController:
    '''
    Lateral control using the Stanley controller

    functions:
        stanley 

    init:
        gain_constant (default=5)
        damping_constant (default=0.5)
    '''


    def __init__(self, gain_constant=3.3, damping_constant=0.6):

        self.gain_constant = gain_constant
        self.damping_constant = damping_constant
        self.previous_steering_angle = 0


    def stanley(self, waypoints, speed, time_step=0):
        '''
        ##### TODO #####
        one step of the stanley controller with damping
        args:
            waypoints (np.array) [2, num_waypoints]
            time_step (int)
            speed (float)
        '''
        # derive orientation error as the angle of the first path segment to the car orientation
        p1 = waypoints[:, 0]
        p2 = waypoints[:, 1]
        deltax, deltay = p2 - p1
        orientation_error = np.arctan2(deltax, deltay)

        # derive cross track error as distance between desired waypoint at spline parameter equal zero ot the car position
        car_position = np.array([48, 0])
        desired_waypoint = waypoints[:, 0]
        error_vec = desired_waypoint - car_position
        cross_track_error = np.linalg.norm(error_vec) * np.sign(error_vec[0])

        #print('Cross track error:', cross_track_error)

        # derive stanley control law
        # prevent division by zero by adding as small epsilon
        stanley_control = orientation_error + np.arctan((self.gain_constant * cross_track_error) / (speed + 1e-6))

        # derive damping term
        
        steering_angle = stanley_control - self.damping_constant * (stanley_control - self.previous_steering_angle)
        # clip to the maximum stering angle (0.4) and rescale the steering action space
        return np.clip(steering_angle, -0.4, 0.4) / 0.4






