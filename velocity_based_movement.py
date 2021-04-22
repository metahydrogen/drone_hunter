####Dependencies###################

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import socket
import exceptions
import argparse
from pymavlink import mavutil

###Function definitions for mission####

##Function to connect script to drone
def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect

    vehicle = connect(connection_string, wait_ready=True)

    return vehicle

##Function to arm the drone and takeoff into the air##
def arm_and_takeoff(aTargetAltitude):
    while not vehicle.is_armable:
        print("Waiting for vehicle to become armable")
        time.sleep(1)

    #Switch vehicle to GUIDED mode and wait for change
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode!="GUIDED":
        print("Waiting for vehicle to enter GUIDED mode")
        time.sleep(1)

    #Arm vehicle once GUIDED mode is confirmed
    vehicle.armed=True
    while vehicle.armed==False:
        print("Waiting for vehicle to become armed.")
        time.sleep(1)

    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print("Current Altitude: %d"%vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*.95:
            break
        time.sleep(1)

    print("Target altitude reached")
    return None
    
def send_local_ned_velocity(u,v,w):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        u, v, w, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)
    vehicle.flush()
    
def condition_yaw(heading):
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        5,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        1,          # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)
    vehicle.flush()


#### Mission################################

vehicle=connectMyCopter()
arm_and_takeoff(10) ##Tell drone to fly 2 meters in the sky

counter=0;
while counter<5:
    send_local_ned_velocity(5,0,0) # moves forward at 5 m/s
    time.sleep(1)
    counter+=1
time.sleep(2)

counter=0;
while counter<5:
    send_local_ned_velocity(0,-5,0) # moves left at 5 m/s
    time.sleep(1)
    counter+=1
    
condition_yaw(90) # yaws 90 degrees clockwise at 5 deg/s (parameters can be modified in condition_yaw function above)
