import rclpy
from simulation_interfaces.srv import GetEntityState, SetEntityState, SpawnEntity, DeleteEntity, SetSimulationState, ResetSimulation, GetEntityInfo, LoadWorld, UnloadWorld, StepSimulation
from simulation_interfaces.msg import Result, SimulationState
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion, Vector3
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from control_msgs.msg import JointTolerance
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray, Header
import argparse
import os
import numpy as np
import time
import logging

DEMO_ASSET_PATH = os.getenv('DEMO_ASSET_PATH')


def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) to a geometry_msgs.msg.Quaternion.

    Returns a Quaternion message with normalized components (w, x, y, z).
    """
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    half_yaw = yaw / 2.0
    w = float(np.cos(half_yaw))
    x = 0.0
    y = 0.0
    z = float(np.sin(half_yaw))
    # Normalize to be safe
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm == 0.0:
        norm = 1.0
    q = Quaternion(w=w / norm, x=x / norm, y=y / norm, z=z / norm)
    return q


def format_entity_name(entity_name, sim_backend):
    """Format entity name according to simulation backend requirements.
    
    For Isaacsim Sim backend, prefixes entity names with forward slash.
    For other backends, returns the name unchanged.
    
    Args:
        entity_name: The entity name to format
        sim_backend: The simulation backend being used
        
    Returns:
        Formatted entity name
    """
    if sim_backend == "isaacsim":
        return f"/{entity_name}" if not entity_name.startswith("/") else entity_name
    return entity_name


def setup_service_clients(node, sim_backend):
    """Create and wait for essential service clients, return them as a tuple."""
    load_world_client = None
    unload_world_client = None
    if sim_backend != "gazebo":
        load_world_client = node.create_client(LoadWorld, 'load_world')
        unload_world_client = node.create_client(UnloadWorld, 'unload_world')

    logger = logging.getLogger(__name__)
    service_prefix_str =''
    if sim_backend == "gazebo":
        service_prefix_str = '/gz_server/'
    set_state_client = node.create_client(SetSimulationState, service_prefix_str + 'set_simulation_state')
    reset_client = node.create_client(ResetSimulation, service_prefix_str + 'reset_simulation')
    spawn_entity_client = node.create_client(SpawnEntity, service_prefix_str + 'spawn_entity')
    get_entity_state_client = node.create_client(GetEntityState, service_prefix_str + 'get_entity_state')
    set_entity_state_client = node.create_client(SetEntityState, service_prefix_str + 'set_entity_state')
    get_entity_info_client = node.create_client(GetEntityInfo, service_prefix_str + 'get_entity_info')
    step_sim_client = node.create_client(StepSimulation, service_prefix_str + 'step_simulation')

    logger.info("Waiting for simulation services...")
    while not set_state_client.wait_for_service(timeout_sec=1.0):
        logger.debug("set_simulation_state service not available, waiting...")
    logger.info("Waiting for simulation services2...")
    if sim_backend != "gazebo":
        while not load_world_client.wait_for_service(timeout_sec=1.0):
            logger.debug("load_world service not available, waiting...")
    logger.info("Waiting for simulation services3...")
    while not spawn_entity_client.wait_for_service(timeout_sec=1.0):
        logger.debug("spawn_entity service not available, waiting...")

    logger.info("All services ready!")
    return (
        set_state_client,
        reset_client,
        load_world_client,
        unload_world_client,
        spawn_entity_client,
        get_entity_state_client,
        set_entity_state_client,
        get_entity_info_client,
        step_sim_client,
    )


def load_warehouse_world(node, load_world_client, uri: str):
    """Load a world by URI and return True on success.

    Args:
        node: rclpy node (for spinning)
        load_world_client: client for LoadWorld
        uri: spawnable URI to load (required)
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading world: %s", uri)
    req = LoadWorld.Request()
    req.uri = uri
    future = load_world_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("World loaded successfully: %s", uri)
        return True
    else:
        logger.error("Failed to load world %s: %s", uri, future.result().result.error_message)
        return False


def unload_world(node, unload_world_client):
    """Call UnloadWorld service and return True on success."""
    logger = logging.getLogger(__name__)
    logger.info("Unloading world...")
    req = UnloadWorld.Request()
    future = unload_world_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() and future.result().result.result == Result.RESULT_OK:
        return True
    else:
        logger.error("Failed to unload world: %s", getattr(future.result().result, 'error_message', 'unknown'))
        return False
    
def spawn_entity(node, spawn_entity_client, name, uri, position, orientation, allow_renaming=True, entity_namespace=None, sim_backend="isaacsim"):
    """Spawn an entity with the given parameters.

    Args:
        node: rclpy node
        spawn_entity_client: client for SpawnEntity
        name: entity name (str)
        uri: asset URI (str)
        position: geometry_msgs.msg.Point
        orientation: geometry_msgs.msg.Quaternion
        allow_renaming: bool (default True)
        entity_namespace: str or None
        sim_backend: simulation backend being used (str)

    Returns:
        True if spawn succeeded, False otherwise.
    """
    logger = logging.getLogger(__name__)
    req = SpawnEntity.Request()
    req.name = format_entity_name(name, sim_backend)
    req.uri = uri
    req.allow_renaming = allow_renaming
    if entity_namespace:
        req.entity_namespace = entity_namespace
    req.initial_pose = PoseStamped()
    req.initial_pose.header.frame_id = "world"
    req.initial_pose.pose.position = position
    req.initial_pose.pose.orientation = orientation
    future = spawn_entity_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("Entity '%s' spawned successfully", name)
        return True
    else:
        logger.error("Failed to spawn entity '%s': %s", name, future.result().result.error_message)
        return False
        
        
def spawn_table_and_get_pose(node, spawn_entity_client, get_entity_state_client, sim_backend):
    """Spawn the warehouse table and return (x,y,z) or (None,None,None) on failure."""
    logger = logging.getLogger(__name__)
    logger.info("Spawning warehouse table...")
    position = Point(x=-1.0, y=-1.5, z=1.19)
    orientation = yaw_to_quaternion(1.5708)
    success = spawn_entity(
        node,
        spawn_entity_client,
        name="warehouse_table",
        uri=ACTIVE_TABLE_URI,
        position=position,
        orientation=orientation,
        allow_renaming=False,
        entity_namespace=None,
        sim_backend=sim_backend,
    )
    if not success:
        logger.error("Failed to spawn table")
        return (None, None, None)
    else:
        logger.info("Table spawned successfully")

    time.sleep(0.5)

    # Get table state for relative positioning
    logger.info("Getting table state for relative positioning...")
    req = GetEntityState.Request()
    req.entity = format_entity_name("warehouse_table", sim_backend)
    future = get_entity_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() and future.result().result.result == Result.RESULT_OK:
        table_state = future.result().state
        table_x = table_state.pose.position.x
        table_y = table_state.pose.position.y
        table_z = table_state.pose.position.z
        logger.info("Table found at position: (%.2f, %.2f, %.2f)", table_x, table_y, table_z)
        return (table_x, table_y, table_z)
    else:
        logger.error("Failed to get table state - cannot proceed with relative positioning")
        logger.error("Error: %s", future.result().result.error_message)
        return (None, None, None)


def spawn_cubes_around_table(node, spawn_entity_client, table_x, table_y, table_z, sim_backend):
    """Spawn cubes around the table using the provided table position."""
    logger = logging.getLogger(__name__)
    logger.info("Spawning cubes around table...")
    cube_configs = [
        ((table_x + 0.2, table_y + 0.2, table_z + 0.1), "blue"),
        ((table_x - 0.2, table_y + 0.2, table_z + 0.1), "red"),
        ((table_x + 0.2, table_y - 0.2, table_z + 0.1), "red"),
        ((table_x - 0.2, table_y - 0.2, table_z + 0.1), "blue"),
        ((table_x, table_y, table_z + 0.1), "blue"),
        ((table_x + 0.1, table_y, table_z + 0.1), "red"),
        ((table_x - 0.1, table_y, table_z + 0.1), "red"),
    ]

    for i, (pos, color) in enumerate(cube_configs):
        position_vec = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        orientation_quat = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        if color == "blue":
            uri = ACTIVE_BLUE_CUBE_URI
        else:
            uri = ACTIVE_RED_CUBE_URI
        success = spawn_entity(
            node,
            spawn_entity_client,
            name=f"{color}_cube_{i}",
            uri=uri,
            position=position_vec,
            orientation=orientation_quat,
            allow_renaming=True,
            entity_namespace=None,
            sim_backend=sim_backend,
        )
        if success:
            logger.info("%s cube spawned successfully at %s", color.capitalize(), pos)
        else:
            logger.error("Failed to spawn %s cube at %s", color, pos)

def move_cubes_and_step_sim(node, set_state_client, set_entity_state_client, step_sim_client, table_x, table_y, table_z, sim_backend):
    """Moving cubes above the table and stepping simulation for 1 second."""
    logger = logging.getLogger(__name__)
    logger.info("Pausing, moving cubes above the table, and stepping simulation...")

    paused = set_simulation_state(node, set_state_client, SimulationState.STATE_PAUSED)
    if paused:
        logger.info("Simulation paused successfully")
    else:
        logger.error("Failed to paused simulation")

    time.sleep(0.5)

    cube_configs = [
        ((table_x + 0.1, table_y + 0.1, table_z + 0.1), "blue"),
        ((table_x - 0.1, table_y + 0.1, table_z + 0.1), "red"),
        ((table_x + 0.1, table_y - 0.1, table_z + 0.1), "red"),
        ((table_x - 0.1, table_y - 0.1, table_z + 0.1), "blue"),
        ((table_x, table_y, table_z + 0.1), "blue"),
        ((table_x + 0.2, table_y, table_z + 0.1), "red"),
        ((table_x - 0.2, table_y, table_z + 0.1), "red"),
    ]

    for i, (pos, color) in enumerate(cube_configs):
        if color == "blue":
            uri = ACTIVE_BLUE_CUBE_URI
        else:
            uri = ACTIVE_RED_CUBE_URI

        z_pos = pos[2] + i * 0.05
        move_entity_to_location(node, set_entity_state_client, f'{color}_cube_{i}', pos[0], pos[1], z_pos, 0.0, sim_backend)

    time.sleep(0.5)

    req = StepSimulation.Request()
    req.steps = 100
    future = step_sim_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("Successfully stepped simulation")
    else:
        logger.info("Failed to step simulation")

def spawn_dingo(node, spawn_entity_client, sim_backend):
    """Spawn the Dingo robot and return a cmd_vel publisher or None."""
    logger = logging.getLogger(__name__)
    logger.info("Spawning Dingo robot...")
    position = Point(x=-4.0, y=-3.0, z=0.0)
    orientation = yaw_to_quaternion(0.0)
    success = spawn_entity(
        node,
        spawn_entity_client,
        name="dingo_robot",
        uri=ACTIVE_DINGO_URI,
        position=position,
        orientation=orientation,
        allow_renaming=False,
        entity_namespace="dingo",
        sim_backend=sim_backend,
    )
    if success:
        logger.info("Dingo robot spawned successfully")
        # Create cmd_vel publisher for dingo
        return node.create_publisher(Twist, "/dingo/cmd_vel", 10)
    else:
        logger.error("Failed to spawn Dingo robot")
        return None


def spawn_ur10(node, spawn_entity_client, table_x, table_y, table_z, sim_backend):
    """Spawn the UR10 robot on the table using the provided table position."""
    logger = logging.getLogger(__name__)
    logger.info("Spawning UR10 robot...")
    position = Point(x=float(table_x), y=float(table_y - 0.64), z=float(table_z))
    orientation = yaw_to_quaternion(1.5708)
    success = spawn_entity(
        node,
        spawn_entity_client,
        name="ur10_robot",
        uri=ACTIVE_UR10_URI,
        position=position,
        orientation=orientation,
        allow_renaming=False,
        entity_namespace="ur10",
        sim_backend=sim_backend,
    )
    if success:
        logger.info("UR10 robot spawned successfully")
    else:
        logger.error("Failed to spawn UR10 robot")


def spawn_obstacle_boxes(node, spawn_entity_client, box_positions, sim_backend, loop_iteration):
    """Spawn obstacle boxes for the provided box_positions list.

    box_positions should be an iterable of tuples: (x, y, z, yaw)
    loop_iteration: current loop iteration number to include in box names
    
    Returns:
        List of successfully spawned box names
    """
    logger = logging.getLogger(__name__)
    spawned_boxes = []
    for i, (box_x, box_y, box_z, box_yaw) in enumerate(box_positions):
        position = Point(x=float(box_x), y=float(box_y), z=float(box_z))
        orientation = yaw_to_quaternion(box_yaw)
        box_name = f"obstacle_box_{loop_iteration}_{i}"
        success = spawn_entity(
            node,
            spawn_entity_client,
            name=box_name,
            uri=ACTIVE_CARDBOARD_URI,
            position=position,
            orientation=orientation,
            allow_renaming=True,
            entity_namespace=None,
            sim_backend=sim_backend,
        )
        if success:
            spawned_boxes.append(box_name)
            logger.info("Obstacle box %s spawned at (%.2f, %.2f)", box_name, box_x, box_y)
        else:
            logger.error("Failed to spawn obstacle box %d", i + 1)
    
    return spawned_boxes
            

def move_dingo_towards_table(node, get_entity_state_client, dingo_cmd_vel_pub, table_x, table_y, sim_backend):
    """Query Dingo state and publish cmd_vel to move it towards the table if far away."""
    logger = logging.getLogger(__name__)
    logger.info("Moving Dingo robot towards table...")
    req = GetEntityState.Request()
    req.entity = format_entity_name("dingo_robot", sim_backend)
    future = get_entity_state_client.call_async(req)

    while not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.01)

    if future.result() and future.result().result.result == Result.RESULT_OK:
        dingo_state = future.result().state
        dingo_x = dingo_state.pose.position.x
        dingo_y = dingo_state.pose.position.y
        dx = table_x - dingo_x
        dy = table_y - dingo_y
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > 1.5:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.3
            cmd_vel.angular.z = np.arctan2(dy, dx) * 0.5
            for _ in range(20):
                dingo_cmd_vel_pub.publish(cmd_vel)
                time.sleep(0.1)
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            for _ in range(5):
                dingo_cmd_vel_pub.publish(cmd_vel)
                time.sleep(0.1)

        logger.info("Dingo robot moved towards table (distance: %.2fm)", distance)
    else:
        logger.error("Failed to query Dingo state: %s", future.result().result.error_message)


def push_box(node, get_entity_state_client, set_entity_state_client, dingo_cmd_vel_pub, box_name, sim_backend, push_direction, box_initial_pos=None):
    """Push boxes: position Dingo behind box based on direction, use cmd_vel to push box 0.5 meters forward in the specified direction, check progress continuously.
    
    Args:
        node: rclpy node
        get_entity_state_client: client for GetEntityState
        set_entity_state_client: client for SetEntityState
        dingo_cmd_vel_pub: publisher for Dingo cmd_vel
        box_name: name of the box to push
        sim_backend: simulation backend being used
        push_direction: direction to push the box ("+X", "-X", "+Y", "-Y")
        box_initial_pos: initial box position (x, y, z) tuple, if None will get current position
        
    Returns:
        tuple: (box_moved_distance_in_direction, box_current_pos) or (None, None) if failed
    """
    logger = logging.getLogger(__name__)
    
    # Get current box state
    req = GetEntityState.Request()
    # For Isaacsim Sim, use child prim path of cardboard box for state retrieval
    if sim_backend == "isaacsim":
        req.entity = format_entity_name(f"{box_name}/SM_CardBoxA_02", sim_backend)
    else:
        req.entity = format_entity_name(box_name, sim_backend)
    future = get_entity_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    
    if not (future.result() and future.result().result.result == Result.RESULT_OK):
        logger.error("Failed to get box state for %s: %s", box_name, future.result().result.error_message)
        return None, None
        
    box_state = future.result().state
    current_box_x = box_state.pose.position.x
    current_box_y = box_state.pose.position.y
    current_box_z = box_state.pose.position.z
    current_box_pos = (current_box_x, current_box_y, current_box_z)
    
    # If this is the first time, store initial position and setup
    if box_initial_pos is None:
        box_initial_pos = current_box_pos
        logger.info("Initial box position for %s: (%.2f, %.2f, %.2f)", box_name, current_box_x, current_box_y, current_box_z)
        
        # Position Dingo behind the box based on push direction (1.5 meters away)
        if push_direction == "+X":
            dingo_x = current_box_x - 0.75  # Behind the box to push towards +X
            dingo_y = current_box_y
            dingo_yaw = 0.0  # Facing +X direction (east)
        elif push_direction == "-X":
            dingo_x = current_box_x + 0.75  # Behind the box to push towards -X  
            dingo_y = current_box_y
            dingo_yaw = 3.14159  # Facing -X direction (west)
        elif push_direction == "+Y":
            dingo_x = current_box_x
            dingo_y = current_box_y - 0.75  # Behind the box to push towards +Y
            dingo_yaw = 1.5708  # Facing +Y direction (north)
        elif push_direction == "-Y":
            dingo_x = current_box_x  
            dingo_y = current_box_y + 0.75  # Behind the box to push towards -Y
            dingo_yaw = -1.5708  # Facing -Y direction (south)
        else:
            logger.error("Invalid push direction: %s. Must be +X, -X, +Y, or -Y", push_direction)
            return None, None
            
        dingo_z = 0.05
        
        if sim_backend == "isaacsim":
            dingo_entity_name = "dingo_robot/base_link"
        else:
            dingo_entity_name = "dingo_robot"
        success = move_entity_to_location(node, set_entity_state_client, dingo_entity_name, dingo_x, dingo_y, dingo_z, dingo_yaw, sim_backend)
        if success:
            logger.info("Positioned Dingo behind box %s at (%.2f, %.2f) to push in %s direction", box_name, dingo_x, dingo_y, push_direction)
        else:
            logger.error("Failed to position Dingo behind box %s", box_name)
            return None, None
            
        time.sleep(1.0)  # Wait a moment for positioning to settle
        
        # Start pushing and monitor continuously until target is reached
        logger.info("Starting continuous box pushing with cmd_vel in %s direction...", push_direction)
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2  # Move forward (Dingo orientation determines the actual push direction)
        cmd_vel.angular.z = 0.0
        
        # Continuous pushing loop - monitor box position until 1.0m moved
        max_iterations = 500  # Safety limit (50 seconds at 0.1s intervals)
        iteration = 0
        
        while iteration < max_iterations:
            # Publish cmd_vel to keep pushing
            if dingo_cmd_vel_pub:
                dingo_cmd_vel_pub.publish(cmd_vel)
            
            # Check box position every few iterations to avoid overwhelming the system
            if iteration % 5 == 0:  # Check every 0.5 seconds
                # Get updated box position
                req = GetEntityState.Request()
                # For Isaacsim Sim, use child prim path of cardboard box for state retrieval
                if sim_backend == "isaacsim":
                    req.entity = format_entity_name(f"{box_name}/SM_CardBoxA_02", sim_backend)
                else:
                    req.entity = format_entity_name(box_name, sim_backend)
                future = get_entity_state_client.call_async(req)
                rclpy.spin_until_future_complete(node, future)
                
                if future.result() and future.result().result.result == Result.RESULT_OK:
                    box_state = future.result().state
                    current_box_x = box_state.pose.position.x
                    current_box_y = box_state.pose.position.y
                    current_box_z = box_state.pose.position.z
                    current_box_pos = (current_box_x, current_box_y, current_box_z)
                    
                    # Calculate distance moved in the specified direction
                    if push_direction == "+X":
                        displacement = current_box_x - box_initial_pos[0]  # Positive when moved in +X direction
                    elif push_direction == "-X":
                        displacement = box_initial_pos[0] - current_box_x  # Positive when moved in -X direction
                    elif push_direction == "+Y":
                        displacement = current_box_y - box_initial_pos[1]  # Positive when moved in +Y direction
                    elif push_direction == "-Y":
                        displacement = box_initial_pos[1] - current_box_y  # Positive when moved in -Y direction
                    else:
                        displacement = 0
                    
                    # Calculate the distance the box has moved in the intended direction so we know when to stop pushing.
                    distance_moved = max(0, displacement)
                    logger.info(
                        "Box %s has moved %.2f meters out of 1.0 meter goal in the %s direction", 
                        box_name, distance_moved, push_direction
                    )
                    
                    # Check if target has moved 0.5 meters
                    if distance_moved >= 0.5:
                        logger.info("Target reached! Box %s moved %.2f meters. Stopping Dingo.", box_name, distance_moved)
                        
                        # Stop Dingo cmd_vel
                        if dingo_cmd_vel_pub:
                            stop_cmd = Twist()
                            stop_cmd.linear.x = 0.0
                            stop_cmd.angular.z = 0.0
                            for _ in range(10):
                                dingo_cmd_vel_pub.publish(stop_cmd)
                                time.sleep(0.1)
                        
                        return distance_moved, current_box_pos
                else:
                    logger.error("Failed to get updated box state during pushing")
            
            time.sleep(0.1)  # 10Hz update rate
            iteration += 1
        
        # Safety timeout reached
        logger.warning("Box pushing timeout reached after %.1f seconds", max_iterations * 0.1)
        if dingo_cmd_vel_pub:
            stop_cmd = Twist()
            stop_cmd.linear.x = 0.0
            stop_cmd.angular.z = 0.0
            for _ in range(10):
                dingo_cmd_vel_pub.publish(stop_cmd)
                time.sleep(0.1)
        
        return distance_moved if 'distance_moved' in locals() else 0.0, current_box_pos
    
    # If box_initial_pos is provided, this means pushing is already complete
    if push_direction == "+X":
        displacement = current_box_x - box_initial_pos[0]  # Positive when moved in +X direction
    elif push_direction == "-X":
        displacement = box_initial_pos[0] - current_box_x  # Positive when moved in -X direction
    elif push_direction == "+Y":
        displacement = current_box_y - box_initial_pos[1]  # Positive when moved in +Y direction
    elif push_direction == "-Y":
        displacement = box_initial_pos[1] - current_box_y  # Positive when moved in -Y direction
    else:
        displacement = 0
    
    distance_moved = max(0, displacement)  # Only count movement in the specified direction
    return distance_moved, current_box_pos


def move_ur10_joints(node, loop_iteration, sim_backend):
    """Publish joint positions for UR10 for the given iteration."""
    logger = logging.getLogger(__name__)
    logger.info("Moving UR10 robot joints...")
    joint_positions_by_iteration = [
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        [0.7854, -1.0472, 0.7854, -0.7854, -1.5708, 0.5236],
        [-0.5236, -2.0944, 2.0944, -2.0944, -1.5708, -0.7854],
    ]
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    positions = joint_positions_by_iteration[loop_iteration]
    if sim_backend == "isaacsim":
        move_ur10_joints_topic(node, joint_names, positions)
    elif sim_backend == "o3de":
        # o3de adds namespace to joint names
        joint_names = [f"ur10/{name}" for name in joint_names]
        move_ur10_joints_action(node, joint_names, positions)
    elif sim_backend == "gazebo":
        move_ur10_joint_array_topic(node, joint_names, positions)
    else:
        logger.error("Unknown simulation backend: %s", sim_backend)


def move_ur10_joints_topic(node, joint_names, positions):
    """Move UR10 joints using JointState publisher."""
    logger = logging.getLogger(__name__)
    ur10_joint_pub = node.create_publisher(JointState, "/ur10/joint_commands", 10)
    joint_cmd = JointState()
    joint_cmd.header = Header()
    joint_cmd.header.stamp = node.get_clock().now().to_msg()
    joint_cmd.name = joint_names
    joint_cmd.position = positions
    for _ in range(10):
        ur10_joint_pub.publish(joint_cmd)
        time.sleep(0.2)
    logger.info("UR10 joint positions updated (topic)")

def move_ur10_joint_array_topic(node, joint_names, positions):
    """Move UR10 joints using Float64MultiArray publisher."""
    logger = logging.getLogger(__name__)
    ur10_joint_pub = node.create_publisher(Float64MultiArray, "/ur10/joint_commands", 10)
    joint_cmd = Float64MultiArray()
    joint_cmd.data = positions
    for _ in range(10):
        ur10_joint_pub.publish(joint_cmd)
        time.sleep(0.2)
    logger.info("UR10 joint positions updated (topic)")

def move_ur10_joints_action(node, joint_names, positions):
    """Move UR10 joints using FollowJointTrajectory action."""
    logger = logging.getLogger(__name__)
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    action_client = ActionClient(node, FollowJointTrajectory, "/ur10/joint_trajectory_controller/follow_joint_trajectory")
    while not action_client.wait_for_server(timeout_sec=1.0):
        logging.info("Waiting for action server...")
    
    goal_msg = FollowJointTrajectory.Goal()
    trajectory = JointTrajectory()
    trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * len(joint_names)
    point.effort = [0.0] * len(joint_names)
    point.time_from_start = Duration(sec=1, nanosec=0)
    trajectory.points.append(point)
    goal_tolerance = [JointTolerance(position=0.01) for _ in range(2)]
    goal_msg = FollowJointTrajectory.Goal()
    goal_msg.trajectory = trajectory
    goal_msg.goal_tolerance = goal_tolerance

    future = action_client.send_goal_async(goal_msg)
    rclpy.spin_until_future_complete(node, future)
    goal_handle = future.result()
    if not goal_handle.accepted:
        logging.error("Goal rejected")
        return

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    logger.info("UR10 trajectory executed (action): %s", result)
    
        
def move_entity_to_location(node, set_entity_state_client, entity, target_x, target_y, target_z, target_yaw=1.5708, sim_backend="isaacsim"):
    """Set an entity's pose (position + orientation) via SetEntityState service.

    Args:
        node: rclpy node
        set_entity_state_client: client for SetEntityState
        entity: str name of the entity to move
        target_x, target_y: floats for desired position
        target_yaw: yaw angle in radians (default 1.5708)
        sim_backend: simulation backend being used (str)
    """
    logger = logging.getLogger(__name__)
    logger.info("Moving '%s' to a specific location...", entity)
    from simulation_interfaces.msg import EntityState
    req = SetEntityState.Request()
    req.entity = format_entity_name(entity, sim_backend)
    state = EntityState()
    state.pose = Pose()
    state.pose.position = Point(x=float(target_x), y=float(target_y), z=float(target_z))
    quat = yaw_to_quaternion(target_yaw)
    state.pose.orientation = quat
    state.twist = Twist()
    state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
    state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
    req.state = state
    future = set_entity_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("%s moved to new position: (%.2f, %.2f, %.2f, yaw: %.2f)", entity, target_x, target_y, target_z, target_yaw)
        return True
    else:
        logger.error("Failed to move %s to new position: %s", entity, future.result().result.error_message)
        return False


def set_simulation_state(node, set_state_client, state):
    """Set the simulation state to the given state enum value. Returns True on success."""
    logger = logging.getLogger(__name__)
    logger.info("Setting simulation state to %d...", state)
    req = SetSimulationState.Request()
    req.state.state = state
    future = set_state_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)

    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("Simulation state set successfully to %d", state)
        return True
    else:
        logger.error("Failed to set simulation state: %s", future.result().result.error_message)
        return False
    

def run_simulation_loop(
    node,
    set_state_client,
    spawn_entity_client,
    get_entity_state_client,
    set_entity_state_client,
    dingo_cmd_vel_pub,
    table_x,
    table_y,
    table_z,
    sim_backend,
):
    """Run the main simulation loop which spawns obstacle boxes, moves robots and handles timing."""
    logger = logging.getLogger(__name__)
    # Start simulation
    if not set_simulation_state(node, set_state_client, SimulationState.STATE_PLAYING):
        return False
    time.sleep(2.0)

    # Hardcoded box selection and push directions for each iteration
    box_push_configs = [
        ("obstacle_box_0_1", "-X"),  # Iteration 0: push obstacle_box_0_1 in -X direction
        ("obstacle_box_1_1", "+Y"),  # Iteration 1: push obstacle_box_1_1 in +Y direction  
        ("obstacle_box_2_0", "+X"),  # Iteration 2: push obstacle_box_2_1 in -Y direction
    ]
    
    # Track box pushing across iterations
    selected_box_name = None
    selected_direction = None
    box_initial_position = None
    box_movement_complete = False

    # Main simulation loop
    for loop_iteration in range(3):
        logger.info("=== Loop iteration %d ===", loop_iteration + 1)

        # Get updated table state (in case it moved)
        logger.info("Getting table state...")
        req = GetEntityState.Request()
        req.entity = format_entity_name("warehouse_table", sim_backend)
        future = get_entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)

        if future.result() and future.result().result.result == Result.RESULT_OK:
            table_state = future.result().state
            table_x = table_state.pose.position.x
            table_y = table_state.pose.position.y
            table_z = table_state.pose.position.z
            logger.info("Table position: (%.2f, %.2f, %.2f)", table_x, table_y, table_z)
        else:
            logger.error("Failed to get table state - ending simulation loop")
            logger.error("Error: %s", future.result().result.error_message)
            break

        # Spawn cardboard boxes at different positions based on loop iteration
        logger.info("Spawning obstacle boxes near table (iteration %d)", loop_iteration + 1)

        box_positions_by_iteration = [
            [
                (table_x - 1.5, table_y - 1.0, 0.0, 0.0),
                (table_x + 0.5, table_y - 2.0, 0.0, 1.5708),
                (table_x - 0.8, table_y - 1.8, 0.0, 0.7854),
            ],
            [
                (table_x - 2.0, table_y + 0.5, 0.0, 1.5708),
                (table_x + 1.5, table_y - 0.5, 0.0, 0.0),
                (table_x + 1.8, table_y + 1.2, 0.0, 0.7854),
            ],
            [
                (table_x - 0.8, table_y + 1.8, 0.0, 0.7854),
                (table_x + 0.3, table_y + 1.5, 0.0, 1.5708),
                (table_x - 1.2, table_y + 0.8, 0.0, 0.0),
            ],
        ]

        box_positions = box_positions_by_iteration[loop_iteration]
        spawned_boxes = spawn_obstacle_boxes(node, spawn_entity_client, box_positions, sim_backend, loop_iteration)
        time.sleep(0.5)        

        # Select box and direction based on current iteration
        if spawned_boxes and loop_iteration < len(box_push_configs):
            target_box, target_direction = box_push_configs[loop_iteration]
            
            if target_box in spawned_boxes:
                current_iteration_box = target_box
                current_iteration_direction = target_direction
                logger.info("Selected box %s for pushing in %s direction (iteration %d)", current_iteration_box, current_iteration_direction, loop_iteration + 1)
            else:
                logger.warning("%s not found, falling back to first available box", target_box)
                current_iteration_box = spawned_boxes[0]
                current_iteration_direction = target_direction  # Keep the planned direction
                logger.info("Selected fallback box %s for pushing in %s direction", current_iteration_box, current_iteration_direction)
            
            # If this is a new box or first time, set it as selected and reset pushing state
            if current_iteration_box != selected_box_name:
                selected_box_name = current_iteration_box
                selected_direction = current_iteration_direction
                box_initial_position = None
                box_movement_complete = False

        # Handle box pushing instead of moving towards table (only when new box selected)
        if selected_box_name and not box_movement_complete and box_initial_position is None:
            logger.info("Starting box pushing for %s in %s direction", selected_box_name, selected_direction)
            distance_moved, current_pos = push_box(
                node, get_entity_state_client, set_entity_state_client, dingo_cmd_vel_pub,
                selected_box_name, sim_backend, selected_direction, box_initial_position
            )
            
            if distance_moved is not None:
                box_initial_position = current_pos
                
                # Mark as complete if box has been pushed 0.5 meters
                if distance_moved >= 0.5:
                    box_movement_complete = True
                    logger.info("Box pushing completed! Box moved %.2f meters.", distance_moved)
                else:
                    # This shouldn't happen with the new continuous monitoring, but just in case
                    logger.warning("Box pushing ended prematurely with only %.2f meters moved", distance_moved)
                    box_movement_complete = True
            else:
                logger.error("Failed to push boxes")
                box_movement_complete = True  # Mark as complete to avoid retrying

        move_ur10_joints(node, loop_iteration, sim_backend)
        time.sleep(0.1)

        # Only move Dingo to specific location if box movement is complete
        if box_movement_complete:
            logger.info("Box movement complete, positioning Dingo at final location")
            new_x = table_x - 3.0
            new_y = table_y - 2.5
            new_z = 0.0
            new_yaw = 1.5708
            # Use "dingo_robot/base_link" for Isaacsim, otherwise "dingo_robot"
            if sim_backend == "isaacsim":
                dingo_entity_name = "dingo_robot/base_link"
            else:
                dingo_entity_name = "dingo_robot"
            move_entity_to_location(node, set_entity_state_client, dingo_entity_name, new_x, new_y, new_z, new_yaw, sim_backend)
        
        time.sleep(0.1)

    logger.info("Simulation loop completed!")

    return True


def main():
    # Configure logging early with a compact, readable format (omit module name)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Parse command-line args to allow runtime selection of asset backend
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-backend", choices=["isaacsim", "o3de", "gazebo"], default="isaacsim",
                        help="Choose which asset backend to use (isaacsim, o3de or gazebo).")
    args, unknown = parser.parse_known_args()

    # Initialize ROS client library
    rclpy.init()

    # If Isaacsim backend requested, map ACTIVE URIs to USD files under DEMO_ASSET_PATH
    if args.sim_backend == "isaacsim":
        if not DEMO_ASSET_PATH:
            raise RuntimeError("DEMO_ASSET_PATH must be set to use IsaacSim asset backend")
        # Override ACTIVE_* URIs to point to USD files
        global ACTIVE_TABLE_URI, ACTIVE_BLUE_CUBE_URI, ACTIVE_RED_CUBE_URI, ACTIVE_DINGO_URI, ACTIVE_UR10_URI, ACTIVE_CARDBOARD_URI, ACTIVE_WORLD_URI
        ACTIVE_WORLD_URI = os.path.join(DEMO_ASSET_PATH, "Collected_warehouse_with_forklifts/warehouse_with_forklifts.usd")
        ACTIVE_TABLE_URI = os.path.join(DEMO_ASSET_PATH, "thor_table/thor_table.usd")
        ACTIVE_BLUE_CUBE_URI = os.path.join(DEMO_ASSET_PATH, "Collected_blue_block/blue_block.usd")
        ACTIVE_RED_CUBE_URI = os.path.join(DEMO_ASSET_PATH, "Collected_red_block/red_block.usd")
        ACTIVE_DINGO_URI = os.path.join(DEMO_ASSET_PATH, "dingo/dingo_ROS.usd")
        ACTIVE_UR10_URI = os.path.join(DEMO_ASSET_PATH, "Collected_ur10e_robotiq2f-140_ROS/ur10e_robotiq2f-140_ROS.usd")
        ACTIVE_CARDBOARD_URI = os.path.join(DEMO_ASSET_PATH, "Collected_warehouse_with_forklifts/Props/SM_CardBoxA_02.usd")
    elif args.sim_backend == "o3de":
        ACTIVE_WORLD_URI = "levels/warehouse/warehouse.spawnable"
        ACTIVE_TABLE_URI = "product_asset:///assets/props/thortable/thortable.spawnable"
        ACTIVE_BLUE_CUBE_URI = "product_asset:///assets/props/collectedblocks/basicblock_blue.spawnable"
        ACTIVE_RED_CUBE_URI = "product_asset:///assets/props/collectedblocks/basicblock_red.spawnable"
        ACTIVE_DINGO_URI = "product_asset:///assets/dingo/dingo-d.spawnable"
        ACTIVE_UR10_URI = "product_asset:///prefabs/ur10-with-fingers.spawnable"
        ACTIVE_CARDBOARD_URI = "product_asset:///assets/props/sm_cardboxa_02.spawnable"
    elif args.sim_backend == "gazebo":
        ACTIVE_TABLE_URI = os.path.join(DEMO_ASSET_PATH, "thor_table")
        ACTIVE_BLUE_CUBE_URI = os.path.join(DEMO_ASSET_PATH, "blue_block")
        ACTIVE_RED_CUBE_URI = os.path.join(DEMO_ASSET_PATH, "red_block")
        ACTIVE_DINGO_URI = os.path.join(DEMO_ASSET_PATH, "dingo_d")
        ACTIVE_UR10_URI = os.path.join(DEMO_ASSET_PATH, "ur10")
        ACTIVE_CARDBOARD_URI = os.path.join(DEMO_ASSET_PATH, "cardboard_box")
    else:
        raise RuntimeError(f"Unknown simulation backend: {args.sim_backend}")
        
    # Initialize main ROS node
    node = rclpy.create_node("warehouse_simulation")

    # Create publishers for robot control
    dingo_cmd_vel_pub = None
    ur10_joint_pub = None

    try:
        # Setup service clients and wait for critical services
        (
            set_state_client,
            reset_client,
            load_world_client,
            unload_world_client,
            spawn_entity_client,
            get_entity_state_client,
            set_entity_state_client,
            get_entity_info_client,
            step_sim_client,
        ) = setup_service_clients(node, sim_backend=args.sim_backend)

        # Load the warehouse world (explicit URI)
        if args.sim_backend != "gazebo":
            if not load_warehouse_world(node, load_world_client, ACTIVE_WORLD_URI):
                return
            time.sleep(1.0)

        # Spawn table and get its pose
        table_x, table_y, table_z = spawn_table_and_get_pose(node, spawn_entity_client, get_entity_state_client, args.sim_backend)
        if table_x is None:
            return

        # Spawn cubes around the table
        spawn_cubes_around_table(node, spawn_entity_client, table_x, table_y, table_z, args.sim_backend)
        time.sleep(0.5)
        
        # Spawn Dingo robot (returns cmd_vel publisher or None)
        dingo_cmd_vel_pub = spawn_dingo(node, spawn_entity_client, args.sim_backend)
        time.sleep(0.5)

        # Spawn UR10 robot
        spawn_ur10(node, spawn_entity_client, table_x, table_y, table_z, args.sim_backend)
        time.sleep(2.0)

        # Move box and set entity state
        move_cubes_and_step_sim(node, set_state_client, set_entity_state_client, step_sim_client, table_x, table_y, table_z, args.sim_backend)

        time.sleep(1.0)

        # Run the main simulation loop (spawns obstacles, moves robots, stops and unloads)
        success = run_simulation_loop(
            node,
            set_state_client,
            spawn_entity_client,
            get_entity_state_client,
            set_entity_state_client,
            dingo_cmd_vel_pub,
            table_x,
            table_y,
            table_z,
            sim_backend=args.sim_backend,
        )
        
        # Stop simulation
        logger = logging.getLogger(__name__)
        stopped = set_simulation_state(node, set_state_client, SimulationState.STATE_STOPPED)
        if stopped:
            logger.info("Simulation stopped successfully")
        else:
            logger.error("Failed to stop simulation")

        time.sleep(0.5)

        # Unload world
        if args.sim_backend != "gazebo":
            if unload_world(node, unload_world_client):
                logger.info("World unloaded successfully")
            else:
                logger.error("Failed to unload world")
            
        logger.info("Warehouse simulation completed!")
        if not success:
            logging.getLogger(__name__).error("Simulation loop failed or stopped early")
        
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Interrupted! Stopping simulation...")

        # Stop simulation
        set_simulation_state(node, set_state_client, SimulationState.STATE_STOPPED)
        logger.info("Simulation stopped due to interruption!")
        
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
