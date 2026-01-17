import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time


def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 1e-5
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    ##### TODO #####
    Curvature as  the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args: 
        waypoints [2, num_waypoints] !!!!!
    '''

    way_elements = waypoints[:, 1:] - waypoints[:, :-1]
    curv = np.sum(np.asarray([np.inner(way_elements[:, i], way_elements[:, i+1]) / 
                              (np.linalg.norm(way_elements[:, i]) * np.linalg.norm(way_elements[:, i+1])) 
                              for i in range(way_elements.shape[1] - 1)]))

    return curv


def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    waypoints = waypoints.reshape(2,-1)
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return ls_tocenter - weight_curvature * curv


def waypoint_prediction(roadside1_spline, roadside2_spline, time_step=0, num_waypoints=6, way_type = "smooth"):
    '''
    ##### TODO #####
    Predict waypoint via two different methods:
    - center
    - smooth 

    args:
        roadside1_spline
        roadside2_spline
        time_step
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    def _straight_line():
        t = np.linspace(0, num_waypoints - 1, num_waypoints)
        return np.vstack((np.full(num_waypoints, 48.0), t))

    if time_step < 30:
        return _straight_line()

    if way_type == "center":
        ##### TODO #####
     
        # create spline arguments
        t = np.linspace(0, 1, 6)

        # derive roadside points from spline
        roadside1_points = np.array(splev(t, roadside1_spline))
        roadside2_points = np.array(splev(t, roadside2_spline))

        # derive center between corresponding roadside points
        center_points = (roadside1_points + roadside2_points)/2

        # output way_points with shape(2 x Num_waypoints)
        way_points = center_points

        return way_points
    
    elif way_type == "smooth":
        ##### TODO #####

        # create spline arguments
        t = np.linspace(0, 1, 6)

        # derive roadside points from spline
        roadside1_points = np.array(splev(t, roadside1_spline))
        roadside2_points = np.array(splev(t, roadside2_spline))

        # derive center between corresponding roadside points
        way_points_center = (roadside1_points + roadside2_points)/2
        
        # optimization
        way_points = minimize(smoothing_objective, 
                      (way_points_center), 
                      args=way_points_center)["x"]

        return way_points.reshape(2,-1)


def target_speed_prediction(waypoints, time_step=0, num_waypoints_used=5, max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    ##### TODO #####
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        time_step(int)
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)
    
    output:
        target_speed (float)
    '''

    curv = curvature(waypoints[:, :num_waypoints_used])

    target_speed = (max_speed - offset_speed) * np.exp(- exp_constant * np.abs(num_waypoints_used - 2 - curv)) + offset_speed
    print(target_speed)
    return target_speed
