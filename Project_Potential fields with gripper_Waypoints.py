import coppeliasim_zmqremoteapi_client as zmq
import matplotlib.pyplot as plt
import numpy as np
import time

def plot_joint_positions(dh_params, sim, base_pos):
    """
    Plots the joint positions on the x-z axis using the DH parameters.

    Parameters:
    - dh_params: List of DH parameters for the robot.
    - sim: The CoppeliaSim API object.
    - base_handle: The handle for the robot's base in CoppeliaSim.
    """
    # Get all joint positions
    T_prev = np.eye(4)
    joint_positions = [base_pos]

    for i in range(len(dh_params)):
        # Unpack DH parameters
        alpha, a, d, theta, offset = dh_params[i]
        # Compute transformation matrix for the current link
        T = DH_to_T(alpha, a, d, theta, offset)
        # Update cumulative transformation matrix
        T_prev = np.dot(T_prev, T)
        # Get the current joint position
        current_pos = T_prev[:3, 3]
        joint_positions.append(current_pos)

    joint_positions = np.array(joint_positions)

    # Plot the joint positions on the x-z axis
    plt.figure()
    plt.plot(joint_positions[:, 0], joint_positions[:, 2], marker='o', label='Joint Positions')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Joint Positions on X-Z Plane')
    plt.grid(True)
    plt.legend()
    plt.show()

def DH_to_T(alpha, a, d, theta, offset=0):
    """
    Computes the DH Transformation matrix from link i-1 to link i.
    
    Parameters:
    - alpha: The twist angle between the z_{i-1} and z_i axes, in radians.
    - a: The link length, the distance between z_{i-1} and z_i along x_{i-1}.
    - d: The link offset, the displacement along the z_i axis.
    - theta: The joint angle, the rotation about the z_{i-1} axis.
    - offset: An additional rotational offset applied to the joint angle theta.
    
    Returns:
    - T: A 4x4 numpy array representing the transformation matrix.
    """
    # Compute the combined theta with offset
    theta = theta + offset
    
    # Compute the transformation matrix
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])
    
    return T

def compute_midpoint_DH(dh_params, joint_index, base_position):
    """
    Computes the midpoint of a joint in a robot using DH parameters.
    
    Parameters:
    - dh_params: List of DH parameters for the robot, where each entry is a tuple (alpha, a, d, theta, offset).
                 Example: [(alpha1, a1, d1, theta1, offset1), (alpha2, a2, d2, theta2, offset2), ...]
    - joint_index: The index of the joint for which to compute the midpoint (1-based index).
    
    Returns:
    - midpoint: A numpy array [x, y, z] representing the midpoint position in 3D space.
    """
    if joint_index < 1 or joint_index > len(dh_params):
        raise ValueError("joint_index must be between 1 and the number of DH parameter sets.")
    
    # Initialize transformation matrix to identity
    T_prev = np.eye(4)
    positions = [base_position] # Start with the base at (0, 0, 0)
    
    for i in range(joint_index):
        # Unpack DH parameters for the current link
        alpha, a, d, theta, offset = dh_params[i]
        # Compute the transformation matrix for this link
        T = DH_to_T(alpha, a, d/2, theta, offset)
        # Multiply to get the cumulative transformation up to this link
        T_prev = np.dot(T_prev, T)
        # Extract the position of the current joint
        positions.append(T_prev[:3, 3])
    
    # Compute the midpoint between the previous joint and the current joint
    midpoint = (positions[-2] + positions[-1]) / 2
    return midpoint

def get_position(sim, handle):
    # Get the target point position
    pos = sim.getObjectPosition(handle,sim.handle_world)
    return pos

def plot_grid(end_effector1, end_effector2, target):
    # Plot the grid map
    plt.figure(figsize=(8, 6))
    plt.scatter(end_effector1[0], end_effector1[1], color='blue', label="End Effector 1", s=100)
    plt.scatter(end_effector2[0], end_effector2[1], color='green', label="End Effector 2", s=100)
    plt.scatter(target[0], target[1], color='red', label="Target", s=100)

    # Grid properties
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Labels
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    
    # Show plot
    plt.title("Robot End Effectors and Target Location")
    plt.show()

def compute_jacobian(q1, q2, q3, l0, l1, l2, l3):
    """
    This function computes the Jacobian matrix for a 3-DOF robotic manipulator.

    Parameters:
    - q1: Joint angle for the base rotation (rotation about the z-axis).
    - q2: Joint angle between the first and second link (rotation about the y-axis).
    - q3: Joint angle between the second and third link.
    - l0: Base link length (height from ground to first joint, though not used directly here).
    - l1: The length of the first link.
    - l2: The length of the second link.
    - l3: The length of the third link.

    Returns:
    - J: A 3x3 Jacobian matrix, which describes the relationship between joint velocities and end-effector velocities in Cartesian coordinates.
    """
    # Calculate the intermediate variables
    U = l1 - l2 * np.sin(q2) + l3 * np.cos(q2 + q3)
    H = l2 * np.cos(q2) + l3 * np.sin(q2 + q3)
    
    # Compute partial derivatives
    dU_q2 = -l2 * np.cos(q2) - l3 * np.sin(q2 + q3)
    dU_q3 = -l3 * np.sin(q2 + q3)
    dH_q2 = -l2 * np.sin(q2) + l3 * np.cos(q2 + q3)
    dH_q3 = l3 * np.cos(q2 + q3)
    
    # Compute the Jacobian matrix
    J = np.array([
        [-U * np.sin(q1), dU_q2 * np.cos(q1), dU_q3 * np.cos(q1)],
        [-U * np.cos(q1), dU_q2 * np.sin(q1), dU_q3 * np.sin(q1)],
        [0, dH_q2, dH_q3]
    ])
    
    return J

def compute_reppotgrad(point,robot_pos,eps=3, min_thresh = 1, max_thresh=15):
    """
    Computes the gradient of the repulsive potential between a point (obstacle) and the robot.
    
    This function calculates the repulsive potential gradient, which acts as a force pushing the robot 
    away from obstacles. The repulsive gradient is only applied when the robot is within a certain 
    threshold distance from the obstacle. For points farther than `max_thresh` or closer than `min_thresh`, 
    the gradient is zero.
    
    Parameters:
    -----------
    point : numpy array
        The position of the obstacle (x, y).
    
    robot_pos : numpy array
        The current position of the robot (x, y).
    
    eps : float, optional
        Scaling factor for the repulsive gradient (default is 3).
    
    min_thresh : float, optional
        Minimum distance threshold for the repulsive gradient to have an effect (default is 1).
    
    max_thresh : float, optional
        Maximum distance threshold for the repulsive gradient to have an effect (default is 15).
    
    Returns:
    --------
    dU : numpy array
        The repulsive potential gradient vector (dUx, dUy). Returns [0, 0] if the robot is outside the 
        threshold distance.
    """
    d = np.linalg.norm(point-robot_pos)
    
    if min_thresh < d and d < max_thresh:
        dU = - (eps) * (robot_pos - point) * (1/(d**3))
    else:
        dU = 0
        
    return np.array(dU)

def compute_attpotgrad(point,robot_pos,eps1=5, eps2=5, max_thresh=5):
    """
    Computes the gradient of the attractive potential between a point (goal) and the robot.

    This function calculates the attractive potential gradient, which acts as a force pulling the robot 
    toward the goal. The attractive gradient behaves differently based on whether the distance between 
    the robot and the goal is below or above a certain threshold (`max_thresh`).
    
    - When the distance is below the threshold, the gradient is quadratic.
    - When the distance is above the threshold, the gradient is conic.

    Parameters:
    -----------
    point : numpy array
        The position of the goal (x, y).
    
    robot_pos : numpy array
        The current position of the robot (x, y).
    
    eps1 : float, optional
        Scaling factor for the attractive gradient when within the threshold (default is 5).
    
    eps2 : float, optional
        Scaling factor for the attractive gradient when beyond the threshold (default is 5).
    
    max_thresh : float, optional
        Distance threshold where the attractive potential switches from quadratic to conic (default is 5).

    Returns:
    --------
    dU : list
        The attractive potential gradient vector [dUx, dUy].
    """
    
    d = np.linalg.norm(point-robot_pos)
    
    if d<max_thresh:
        dU = eps1 * (robot_pos - point)
    else:
        dU = max_thresh * eps2 * (robot_pos - point) * (1/d)
    
    return np.array(dU)

#%%
if __name__ == '__main__':
    
    client = zmq.RemoteAPIClient()
    sim = client.getObject('sim')
    
    # grab the robot and goal positions
    cube0_handle = sim.getObjectHandle("/object_1")
    cube1_handle = sim.getObjectHandle("/object_0")
    cube_0 = sim.getObjectHandle("/object_1/Dummy")
    cube_1 = sim.getObjectHandle("/object_0/Dummy")
    platform_0 = sim.getObjectHandle("/platform_0/target")
    platform_1 = sim.getObjectHandle("/platform_1/target")
    platform_0_face = sim.getObjectHandle("/platform_0/front_face")
    platform_1_face = sim.getObjectHandle("/platform_1/front_face")
    platform0_face_world = np.array(get_position(sim,platform_0_face))
    platform1_face_world = np.array(get_position(sim,platform_1_face))


    floor1 = sim.getObjectHandle("/Floor/Dummy[0]")
    floor2 = sim.getObjectHandle("/Floor/Dummy[1]")
    floor3 = sim.getObjectHandle("/Floor/Dummy[2]")
    floor4 = sim.getObjectHandle("/Floor/Dummy[3]")

    floor5 = sim.getObjectHandle("/Floor/Dummy[4]")#to be used in robot 0 side

    floor1_world = np.array(get_position(sim, floor1))
    floor2_world = np.array(get_position(sim, floor2))
    floor3_world = np.array(get_position(sim, floor3))
    floor4_world = np.array(get_position(sim, floor4))
    floor5_world = np.array(get_position(sim, floor5))

    waypoint0_0 = sim.getObjectHandle("/waypoint0_0")
    waypoint0_1 = sim.getObjectHandle("/waypoint0_1")
    waypoint0_2 = sim.getObjectHandle("/waypoint0_2")
    waypoint1_0 = sim.getObjectHandle("/waypoint1_0")
    waypoint1_1 = sim.getObjectHandle("/waypoint1_1")
    waypoint1_2 = sim.getObjectHandle("/waypoint1_2")

    waypoint0_0_world = np.array(get_position(sim, waypoint0_0))
    waypoint0_1_world = np.array(get_position(sim, waypoint0_1))
    waypoint0_2_world = np.array(get_position(sim, waypoint0_2))
    waypoint1_0_world = np.array(get_position(sim, waypoint1_0))
    waypoint1_1_world = np.array(get_position(sim, waypoint1_1))
    waypoint1_2_world = np.array(get_position(sim, waypoint1_2))

    robot0_EE = sim.getObjectHandle("/IRB140[0]/connection/RG2/attachPoint/Dummy")
    robot1_EE = sim.getObjectHandle("/IRB140[1]/connection/RG2/attachPoint/Dummy")

    robot0_base = sim.getObjectHandle("/IRB140[0]/robot_origin")
    robot1_base = sim.getObjectHandle("/IRB140[1]/robot_origin")

    robot0_world = np.array(get_position(sim, robot0_EE))
    robot1_world = np.array(get_position(sim, robot1_EE))
    robot0_base_world = np.array(get_position(sim, robot0_base))
    robot1_base_world = np.array(get_position(sim, robot1_base))

    cube0_world = np.array(get_position(sim,cube_0))
    cube1_world = np.array(get_position(sim,cube_1))
    platform0_world = np.array(get_position(sim,platform_0))
    platform1_world = np.array(get_position(sim,platform_1))

    robot_0_link1 = sim.getObjectHandle("/IRB140[0]/joint")
    robot_0_link2 = sim.getObjectHandle("/IRB140[0]/link/joint")
    robot_0_link3 = sim.getObjectHandle("/IRB140[0]/link/joint/link/joint")

    robot_1_link1 = sim.getObjectHandle("/IRB140[1]/joint")
    robot_1_link2 = sim.getObjectHandle("/IRB140[1]/link/joint")
    robot_1_link3 = sim.getObjectHandle("/IRB140[1]/link/joint/link/joint")

    connector0 = sim.getObjectHandle('/IRB140[0]/attachPoint')
    objectSensor0 = sim.getObjectHandle('/IRB140[0]/connection/RG2/Sensor0')
    attachedShape0 = None  # Keeps track of the currently attached object for Robot 0

    connector1 = sim.getObjectHandle('/IRB140[1]/attachPoint')
    objectSensor1 = sim.getObjectHandle('/IRB140[1]/connection/RG2/Sensor1')
    attachedShape1 = None  # Keeps track of the currently attached object for Robot 1

    # loops until both arms are close to the cube
    ##tweak threshold to come closer or further to the cube
    while (np.linalg.norm(robot0_world - cube0_world) > 0.03 or np.linalg.norm(robot1_world - cube1_world) > 0.03):
        # print("1")
        # print(np.linalg.norm(robot0_world - cube0_world))
        # print("2")
        # print(np.linalg.norm(robot1_world - cube1_world))

        # Poll positions of end effectors and target
        cube0_world = np.array(get_position(sim,cube_0))
        cube1_world = np.array(get_position(sim,cube_1))
        robot0_world = np.array(get_position(sim, robot0_EE))
        robot1_world = np.array(get_position(sim, robot1_EE))
        robot0_base_world = np.array(get_position(sim, robot0_base))
        robot1_base_world = np.array(get_position(sim, robot1_base))

        robot0_link1_pos = sim.getJointPosition(robot_0_link1)
        robot0_link2_pos = sim.getJointPosition(robot_0_link2)
        robot0_link3_pos = sim.getJointPosition(robot_0_link3)

        robot1_link1_pos = sim.getJointPosition(robot_1_link1)
        robot1_link2_pos = sim.getJointPosition(robot_1_link2)
        robot1_link3_pos = sim.getJointPosition(robot_1_link3)

        dh_params_0 = [
        (0, 0, 0.352, robot0_link1_pos, 0),  # Link 1
        (np.pi / 2, 0.070, 0, robot0_link2_pos, np.pi / 2),  # Link 2
        (0, 0.360, 0, robot0_link3_pos, 0),   # Link 3
        (np.pi / 2, 0, 0.574, 0, 0)   # Link 4
    ]
        dh_params_1 = [
        (0, 0, 0.352, robot1_link1_pos, 0),  # Link 1
        (np.pi / 2, 0.070, 0, robot1_link2_pos, np.pi / 2),  # Link 2
        (0, 0.360, 0, robot1_link3_pos, 0),   # Link 3
        (np.pi / 2, 0, 0.574, 0, 0)   # Link 4
    ]
        
        robot_0_link1_midpoint = compute_midpoint_DH(dh_params_0, 1, robot0_base_world)
        robot_0_link2_midpoint = compute_midpoint_DH(dh_params_0, 2, robot0_base_world)
        robot_0_link3_midpoint = compute_midpoint_DH(dh_params_0, 3, robot0_base_world)


        robot_1_link1_midpoint = compute_midpoint_DH(dh_params_1, 1, robot1_base_world)
        robot_1_link2_midpoint = compute_midpoint_DH(dh_params_1, 2, robot1_base_world)
        robot_1_link3_midpoint = compute_midpoint_DH(dh_params_1, 3, robot1_base_world)
        # Plot the positions on the grid map
        # plot_grid(robot1_world, robot2_world, goal1_world)
        # plot_joint_positions(dh_params_0, sim, robot0_base_world)
        # plot_joint_positions(dh_params_0, sim, robot0_base_world)

        F_rep_EE0 = compute_reppotgrad(robot1_world, robot0_world)
        F_rep_link1_0 = compute_reppotgrad(robot_0_link1_midpoint, robot0_world)
        F_rep_link2_0 = compute_reppotgrad(robot_0_link2_midpoint, robot0_world)
        F_rep_link3_0 = compute_reppotgrad(robot_0_link3_midpoint, robot0_world)

        F_rep_EE1 = compute_reppotgrad(robot0_world, robot1_world)
        F_rep_link1_1 = compute_reppotgrad(robot_1_link1_midpoint, robot0_world)
        F_rep_link2_1 = compute_reppotgrad(robot_1_link2_midpoint, robot0_world)
        F_rep_link3_1 = compute_reppotgrad(robot_1_link3_midpoint, robot0_world)

        F_att_0 = compute_attpotgrad(cube0_world, robot0_world)
        F_att_1 = compute_attpotgrad(cube1_world, robot1_world)

        J0 = compute_jacobian(robot0_link1_pos, robot0_link2_pos, robot0_link3_pos, 0.352, 0.070, 0.360, 0.574) #account for length of gripper
        J1 = compute_jacobian(robot1_link1_pos, robot1_link2_pos, robot1_link3_pos, 0.352, 0.070, 0.360, 0.574)

        alpha_att = 1.0
        beta_rep = 1.0
        F_total_0 = -alpha_att * F_att_0 - beta_rep * (F_rep_EE0 + F_rep_link1_0 + F_rep_link2_0 + F_rep_link3_0)
        # print(F_total_1)
        F_total_0[1] = 0
        F_total_1 = -alpha_att * F_att_1 - beta_rep * (F_rep_EE1 + F_rep_link1_1 + F_rep_link2_1 + F_rep_link3_1)
        F_total_1[1] = 0
        # plt.figure()
        # plt.quiver(robot0_world[0], robot0_world[2], F_total_0[0], F_total_0[2], color='blue')
        # plt.quiver(robot1_world[0], robot1_world[2], F_total_1[0], F_total_1[2], color='green')
        # plt.show()  # Non-blocking in interactive mode
        

        if np.linalg.norm(robot0_world - cube0_world) < 0.03:
            robot0_joint_vel = np.zeros_like(robot0_joint_vel)
            robot1_joint_vel = np.matmul(J1.transpose(), F_total_1)
        elif np.linalg.norm(robot1_world - cube1_world) < 0.03:
            robot0_joint_vel = np.matmul(J0.transpose(), F_total_0)
            robot1_joint_vel = np.zeros_like(robot1_joint_vel)
        else:
            robot0_joint_vel = np.matmul(J0.transpose(), F_total_0)
            robot1_joint_vel = np.matmul(J1.transpose(), F_total_1)

        max_velocity = 0.1  # Adjust as needed
        robot0_joint_vel = max_velocity * (robot0_joint_vel / np.linalg.norm(robot0_joint_vel))
        robot1_joint_vel = max_velocity * (robot1_joint_vel / np.linalg.norm(robot1_joint_vel))

        sim.setJointTargetVelocity(robot_0_link1, robot0_joint_vel[0])
        sim.setJointTargetVelocity(robot_0_link2, robot0_joint_vel[1])
        sim.setJointTargetVelocity(robot_0_link3, robot0_joint_vel[2])

        sim.setJointTargetVelocity(robot_1_link1, -robot1_joint_vel[0])
        sim.setJointTargetVelocity(robot_1_link2, -robot1_joint_vel[1])
        sim.setJointTargetVelocity(robot_1_link3, -robot1_joint_vel[2])

        # Wait before polling again
        time.sleep(0.01)


    print("Cube reached")

    sim.setJointTargetVelocity(robot_0_link1, 0)
    sim.setJointTargetVelocity(robot_0_link2, 0)
    sim.setJointTargetVelocity(robot_0_link3, 0)

    sim.setJointTargetVelocity(robot_1_link1, 0)
    sim.setJointTargetVelocity(robot_1_link2, 0)
    sim.setJointTargetVelocity(robot_1_link3, 0)

    # Gripper code for Robot 0 with debugging and cube attachment
    while (attachedShape0 is None):
            
        # Attach the detected object to the gripper
        attachedShape0 = cube0_handle
        sim.setObjectParent(attachedShape0, connector0, True)
        sim.writeCustomDataBlock(attachedShape0, 'attached', 'true')  # Mark the object as attached
        
        # Set the local position of the cube relative to the connector
        sim.setObjectPosition(attachedShape0, connector0, [0, 0, 0])
        
        # Confirm attachment
        parent_handle = sim.getObjectParent(attachedShape0)
        parent_name = sim.getObjectName(parent_handle)
        print(f"Robot 0 attached object: {cube_0}, new parent: {parent_name}")


        # Gripper code for Robot 1 with debugging and cube attachment
    while attachedShape1 is None:
        attachedShape1 = cube1_handle
        sim.setObjectParent(attachedShape1, connector1, True)
        sim.writeCustomDataBlock(attachedShape1, 'attached', 'true')  # Mark the object as attached
        
        # Set the local position of the cube relative to the connector
        sim.setObjectPosition(attachedShape1, connector1, [0, 0, 0])
        
        # Confirm attachment
        parent_handle = sim.getObjectParent(attachedShape1)
        parent_name = sim.getObjectName(parent_handle)
        print(f"Robot 1 attached object: {cube_1}, new parent: {parent_name}")
 

    plt.ion()
    fig, ax = plt.subplots()
    quiver_total_0 = None
    quiver_total_1 = None
    quivers_individual_0 = []  # List to hold individual force arrows for Robot 0
    quivers_individual_1 = []  # List to hold individual force arrows for Robot 1
    legend_items = []  # For dynamically adding legends

    waypoints = [(waypoint0_0, waypoint1_0), (waypoint0_1, waypoint1_1), (waypoint0_2, waypoint1_2), (platform_0, platform_1)]
    for waypoint in waypoints:
        while (np.linalg.norm(robot0_world - np.array(get_position(sim,waypoint[0]))) > 0.3 or np.linalg.norm(robot1_world - np.array(get_position(sim,waypoint[1]))) > 0.4):
            # Poll positions of end effectors and target
            waypoint0 = np.array(get_position(sim,waypoint[0]))
            waypoint1 = np.array(get_position(sim,waypoint[1]))
            robot0_world = np.array(get_position(sim, robot0_EE))
            robot1_world = np.array(get_position(sim, robot1_EE))

            robot0_link1_pos = sim.getJointPosition(robot_0_link1)
            robot0_link2_pos = sim.getJointPosition(robot_0_link2)
            robot0_link3_pos = sim.getJointPosition(robot_0_link3)

            robot1_link1_pos = sim.getJointPosition(robot_1_link1)
            robot1_link2_pos = sim.getJointPosition(robot_1_link2)
            robot1_link3_pos = sim.getJointPosition(robot_1_link3)

            dh_params_0 = [
            (0, 0, 0.352, robot0_link1_pos, 0),  # Link 1
            (np.pi / 2, 0.070, 0, robot0_link2_pos, np.pi / 2),  # Link 2
            (0, 0.360, 0, robot0_link3_pos, 0),   # Link 3
            (np.pi / 2, 0, 0.574, 0, 0)   # Link 4
        ]
            dh_params_1 = [
            (0, 0, 0.352, robot1_link1_pos, 0),  # Link 1
            (np.pi / 2, 0.070, 0, robot1_link2_pos, np.pi / 2),  # Link 2
            (0, 0.360, 0, robot1_link3_pos, 0),   # Link 3
            (np.pi / 2, 0, 0.574, 0, 0)   # Link 4
        ]

            robot_0_link1_midpoint = compute_midpoint_DH(dh_params_0, 1, robot0_base_world)
            robot_0_link2_midpoint = compute_midpoint_DH(dh_params_0, 2, robot0_base_world)
            robot_0_link3_midpoint = compute_midpoint_DH(dh_params_0, 3, robot0_base_world)

            
            robot_1_link1_midpoint = compute_midpoint_DH(dh_params_1, 1, robot1_base_world)
            robot_1_link2_midpoint = compute_midpoint_DH(dh_params_1, 2, robot1_base_world)
            robot_1_link3_midpoint = compute_midpoint_DH(dh_params_1, 3, robot1_base_world)
            # plot_joint_positions(dh_params_0, sim, robot0_base_world)
            # plot_joint_positions(dh_params_1, sim, robot1_base_world)

            # F_rep_floor_0 = compute_reppotgrad(floor5_world, robot0_world, min_thresh= 0.01, max_thresh=0.1)
            F_rep_EE0 = compute_reppotgrad(robot1_world, robot0_world, min_thresh= 0.001, max_thresh=0.005)
            F_rep_platform0_face = compute_reppotgrad(platform0_face_world, robot0_world, eps=10, min_thresh= 0, max_thresh=0.2)
            F_rep_base_0 = compute_reppotgrad(robot0_base_world, robot0_world, eps=10,  min_thresh= 0, max_thresh=0.7)
            F_rep_link1_0 = compute_reppotgrad(robot_0_link1_midpoint, robot0_world, eps= 20, min_thresh= 0, max_thresh=0.1)
            F_rep_link2_0 = compute_reppotgrad(robot_0_link2_midpoint, robot0_world, eps= 20, min_thresh= 0, max_thresh=0.01)
            F_rep_link3_0 = compute_reppotgrad(robot_0_link3_midpoint, robot0_world, eps= 20, min_thresh= 0, max_thresh=0.01)

            F_rep_floor1_1 = compute_reppotgrad(floor1_world, robot1_world, eps=0.01, min_thresh= 0.01, max_thresh=0.2)
            # F_rep_floor2_1 = compute_reppotgrad(floor2_world, robot1_world, eps=1, min_thresh= 0.01, max_thresh=0.01)
            # F_rep_floor3_1 = compute_reppotgrad(floor3_world, robot1_world, eps=1, min_thresh= 0.01, max_thresh=0.01)
            # F_rep_floor4_1 = compute_reppotgrad(floor4_world, robot1_world, eps=1, min_thresh= 0.01, max_thresh=0.05)
            F_rep_EE1 = compute_reppotgrad(robot0_world, robot1_world, min_thresh= 0.001, max_thresh=0.005)
            F_rep_platform1_face = compute_reppotgrad(platform1_face_world, robot1_world, min_thresh= 0, max_thresh=0.12)
            F_rep_base_1 = compute_reppotgrad(robot1_base_world, robot1_world, min_thresh= 0, max_thresh=0.1)
            F_rep_link1_1 = compute_reppotgrad(robot_1_link1_midpoint, robot1_world, min_thresh= 0, max_thresh=0.005)
            F_rep_link2_1 = compute_reppotgrad(robot_1_link2_midpoint, robot1_world, min_thresh= 0, max_thresh=0.005)
            F_rep_link3_1 = compute_reppotgrad(robot_1_link3_midpoint, robot1_world, min_thresh= 0, max_thresh=0.005)

            F_att_0 = compute_attpotgrad(waypoint0, robot0_world, max_thresh=0.1, eps2 = 100)
            F_att_1 = compute_attpotgrad(waypoint1, robot1_world, max_thresh=0.1, eps2 = 100)

            J0 = compute_jacobian(robot0_link1_pos, robot0_link2_pos, robot0_link3_pos, 0.352, 0.070, 0.360, 0.574) #account for length of gripper
            J1 = compute_jacobian(robot1_link1_pos, robot1_link2_pos, robot1_link3_pos, 0.352, 0.070, 0.360, 0.574)

            alpha_att = 1.0
            beta_rep = 1.0
            F_total_0 = -alpha_att * F_att_0 - beta_rep * (F_rep_EE0 + F_rep_link1_0 + F_rep_link2_0 + F_rep_link3_0 + F_rep_base_0)
            # print(F_total_1)
            # F_total_0[1] = 0
            F_total_1 = -alpha_att * F_att_1 - beta_rep * (F_rep_EE1 + F_rep_link1_1 + F_rep_link2_1 + F_rep_link3_1 + F_rep_base_1 + F_rep_floor1_1)
            # F_total_1[1] = 0

            if F_total_0.ndim == 0:
                F_total_0 = np.zeros(3)
            if F_total_1.ndim == 0:
                F_total_1 = np.zeros(3)
            F_total_0[1] = 0
            F_total_1[1] = 0

            # Clear existing quivers from the plot
            if quiver_total_0:
                quiver_total_0.remove()
            if quiver_total_1:
                quiver_total_1.remove()
            for q in quivers_individual_0:
                q.remove()
            for q in quivers_individual_1:
                q.remove()
            quivers_individual_0.clear()
            quivers_individual_1.clear()

            # Plot total forces
            quiver_total_0 = ax.quiver(robot0_world[0], robot0_world[2], F_total_0[0], F_total_0[2],
                                    color='blue', scale=1, scale_units='xy', label='Total Force Robot 0')
            quiver_total_1 = ax.quiver(robot1_world[0], robot1_world[2], F_total_1[0], F_total_1[2],
                                    color='green', scale=1, scale_units='xy', label='Total Force Robot 1')

            # Plot individual forces dynamically for Robot 0
            individual_forces_0 = {
                "Attractive Force Robot 0": F_att_0,
                "Repulsive EE Force Robot 0": F_rep_EE0,
                "Repulsive Link1 Force Robot 0": F_rep_link1_0,
                "Repulsive Link2 Force Robot 0": F_rep_link2_0,
                "Repulsive Link3 Force Robot 0": F_rep_link3_0,
                "Repulsive Base Force Robot 0": F_rep_base_0,
                # "Repulsive Face Force Robot 0": F_rep_platform0_face,
                # "Repulsive Floor Force Robot 0": F_rep_floor_0

            }

            for label, F_ind in individual_forces_0.items():
                F_ind = np.array(F_ind)
                if F_ind.ndim == 0:  # Handle scalars
                    F_ind = np.zeros(3)
                F_ind[1] = 0  # Ignore y-axis
                quivers_individual_0.append(ax.quiver(robot0_world[0], robot0_world[2], F_ind[0], F_ind[2],
                                                    color='red', alpha=0.5, scale=1, scale_units='xy', label=label))

            # Plot individual forces dynamically for Robot 1
            individual_forces_1 = {
                "Attractive Force Robot 1": F_att_1,
                "Repulsive EE Force Robot 1": F_rep_EE1,
                "Repulsive Link1 Force Robot 1": F_rep_link1_1,
                "Repulsive Link2 Force Robot 1": F_rep_link2_1,
                "Repulsive Link3 Force Robot 1": F_rep_link3_1,
                "Repulsive Base Force Robot 1": F_rep_base_1,
                # "Repulsive Face Force Robot 1": F_rep_platform1_face,
                "Repulsive Floor1 Force Robot 1": F_rep_floor1_1,
                # "Repulsive Floor2 Force Robot 1": F_rep_floor2_1,
                # "Repulsive Floor3 Force Robot 1": F_rep_floor3_1,
                # "Repulsive Floor4 Force Robot 1": F_rep_floor4_1
            }

            for label, F_ind in individual_forces_1.items():
                F_ind = np.array(F_ind)
                if F_ind.ndim == 0:  # Handle scalars
                    F_ind = np.zeros(3)
                F_ind[1] = 0  # Ignore y-axis
                quivers_individual_1.append(ax.quiver(robot1_world[0], robot1_world[2], F_ind[0], F_ind[2],
                                                    color='orange', alpha=0.5, scale=1, scale_units='xy', label=label))

            # Update plot settings
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_title("Force Vectors Acting on Robot End Effectors")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Z Position")
            ax.grid(True)

            # Display legend only once
            if not ax.get_legend():
                ax.legend(loc="upper left", fontsize="small", ncol=2)

            plt.pause(0.01)

            if np.linalg.norm(robot0_world - platform0_world) < 0.3:
                robot0_joint_vel = np.zeros_like(robot0_joint_vel)
                robot1_joint_vel = np.matmul(J1.transpose(), F_total_1)
                print("Robot 0 reached target")
            elif np.linalg.norm(robot1_world - platform1_world) < 0.3:
                robot0_joint_vel = np.matmul(J0.transpose(), F_total_0)
                robot1_joint_vel = np.zeros_like(robot1_joint_vel)
                print("Robot 1 reached target")
            else:
                robot0_joint_vel = np.matmul(J0.transpose(), F_total_0)
                robot1_joint_vel = np.matmul(J1.transpose(), F_total_1)

            max_velocity = 0.1  # Adjust as needed
            robot0_joint_vel = max_velocity * (robot0_joint_vel / np.linalg.norm(robot0_joint_vel))
            robot1_joint_vel = max_velocity * (robot1_joint_vel / np.linalg.norm(robot1_joint_vel))

            sim.setJointTargetVelocity(robot_0_link1, robot0_joint_vel[0])
            sim.setJointTargetVelocity(robot_0_link2, robot0_joint_vel[1])
            sim.setJointTargetVelocity(robot_0_link3, robot0_joint_vel[2])

            sim.setJointTargetVelocity(robot_1_link1, -robot1_joint_vel[0])
            sim.setJointTargetVelocity(robot_1_link2, -robot1_joint_vel[1])
            sim.setJointTargetVelocity(robot_1_link3, -robot1_joint_vel[2])

            time.sleep(0.01)
            

sim.setJointTargetVelocity(robot_0_link1, 0)
sim.setJointTargetVelocity(robot_0_link2, 0)
sim.setJointTargetVelocity(robot_0_link3, 0)

sim.setJointTargetVelocity(robot_1_link1, 0)
sim.setJointTargetVelocity(robot_1_link2, 0)
sim.setJointTargetVelocity(robot_1_link3, 0)

print("Target reached")
#gripper release
if attachedShape0 is not None:
    sim.setObjectParent(attachedShape0, -1, True)
    sim.writeCustomDataBlock(attachedShape0, 'attached', '')
    print('Robot 0 object detached:', sim.getObjectName(attachedShape0))
    attachedShape0 = None

if attachedShape1 is not None:
    sim.setObjectParent(attachedShape1, -1, True)
    sim.writeCustomDataBlock(attachedShape1, 'attached', '')
    print('Robot 1 object detached:', sim.getObjectName(attachedShape1))
    attachedShape1 = None
