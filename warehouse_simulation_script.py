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
    
def spawn_entity(node, spawn_entity_client, name, uri, position, orientation, allow_renaming=True, entity_namespace=None):
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

    Returns:
        True if spawn succeeded, False otherwise.
    """
    logger = logging.getLogger(__name__)
    req = SpawnEntity.Request()
    req.name = name
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
        
        
def spawn_table_and_get_pose(node, spawn_entity_client, get_entity_state_client):
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
    req.entity = "warehouse_table"
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


def spawn_cubes_around_table(node, spawn_entity_client, table_x, table_y, table_z):
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
        )
        if success:
            logger.info("%s cube spawned successfully at %s", color.capitalize(), pos)
        else:
            logger.error("Failed to spawn %s cube at %s", color, pos)

def move_cubes_and_step_sim(node, set_state_client, set_entity_state_client, step_sim_client, table_x, table_y, table_z):
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
        move_entity_to_location(node, set_entity_state_client, f'{color}_cube_{i}', pos[0], pos[1], z_pos, 0.0)

    time.sleep(0.5)

    req = StepSimulation.Request()
    req.steps = 1000
    future = step_sim_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() and future.result().result.result == Result.RESULT_OK:
        logger.info("Successfully stepped simulation")
    else:
        logger.info("Failed to step simulation")

def spawn_dingo(node, spawn_entity_client):
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
    )
    if success:
        logger.info("Dingo robot spawned successfully")
        # Create cmd_vel publisher for dingo
        return node.create_publisher(Twist, "/dingo/cmd_vel", 10)
    else:
        logger.error("Failed to spawn Dingo robot")
        return None


def spawn_ur10(node, spawn_entity_client, table_x, table_y, table_z):
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
    )
    if success:
        logger.info("UR10 robot spawned successfully")
    else:
        logger.error("Failed to spawn UR10 robot")


def spawn_obstacle_boxes(node, spawn_entity_client, box_positions):
    """Spawn obstacle boxes for the provided box_positions list.

    box_positions should be an iterable of tuples: (x, y, z, yaw)
    """
    logger = logging.getLogger(__name__)
    for i, (box_x, box_y, box_z, box_yaw) in enumerate(box_positions):
        position = Point(x=float(box_x), y=float(box_y), z=float(box_z))
        orientation = yaw_to_quaternion(box_yaw)
        success = spawn_entity(
            node,
            spawn_entity_client,
            name="obstacle_box",
            uri=ACTIVE_CARDBOARD_URI,
            position=position,
            orientation=orientation,
            allow_renaming=True,
            entity_namespace=None,
        )
        if success:
            logger.info("Obstacle box %d spawned at (%.2f, %.2f)", i + 1, box_x, box_y)
        else:
            logger.error("Failed to spawn obstacle box %d", i + 1)
            

def move_dingo_towards_table(node, get_entity_state_client, dingo_cmd_vel_pub, table_x, table_y):
    """Query Dingo state and publish cmd_vel to move it towards the table if far away."""
    logger = logging.getLogger(__name__)
    logger.info("Moving Dingo robot towards table...")
    req = GetEntityState.Request()
    req.entity = "dingo_robot"
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
    if sim_backend == "isaac":
        move_ur10_joints_topic(node, joint_names, positions)
    elif sim_backend == "o3de":
        # o3de adds namespace to joint names
        joint_names = [f"ur10/{name}" for name in joint_names]
        move_ur10_joints_action(node, joint_names, positions)
    if sim_backend == "gazebo":
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
    
        
def move_entity_to_location(node, set_entity_state_client, entity, target_x, target_y, target_z, target_yaw=1.5708):
    """Set an entity's pose (position + orientation) via SetEntityState service.

    Args:
        node: rclpy node
        set_entity_state_client: client for SetEntityState
        entity: str name of the entity to move
        target_x, target_y: floats for desired position
        target_yaw: yaw angle in radians (default 1.5708)
    """
    logger = logging.getLogger(__name__)
    logger.info("Moving '%s' to a specific location...", entity)
    from simulation_interfaces.msg import EntityState
    req = SetEntityState.Request()
    req.entity = entity
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

    # Main simulation loop
    for loop_iteration in range(3):
        logger.info("=== Loop iteration %d ===", loop_iteration + 1)

        # Get updated table state (in case it moved)
        logger.info("Getting table state...")
        req = GetEntityState.Request()
        req.entity = "warehouse_table"
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
        spawn_obstacle_boxes(node, spawn_entity_client, box_positions)
        time.sleep(0.5)        

        # Move the Dingo robot towards the table
        if dingo_cmd_vel_pub:
            move_dingo_towards_table(
                node, get_entity_state_client, dingo_cmd_vel_pub, table_x, table_y
            )

        move_ur10_joints(node, loop_iteration, sim_backend)
        time.sleep(0.1)

        # Move (set) the Dingo to a specific location relative to the table
        new_x = table_x - 3.0
        new_y = table_y - 2.5
        new_z = 0.0
        new_yaw = 1.5708
        move_entity_to_location(node, set_entity_state_client, 'dingo_robot', new_x, new_y, new_z, new_yaw)
        time.sleep(0.1)

    logger.info("Simulation loop completed!")

    return True


def main():
    # Configure logging early with a compact, readable format (omit module name)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # Parse command-line args to allow runtime selection of asset backend
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-backend", choices=["isaac", "o3de", "gazebo"], default="isaac",
                        help="Choose which asset backend to use (isaac, o3de or gazebo).")
    args, unknown = parser.parse_known_args()

    # Initialize ROS client library
    rclpy.init()

    # If Isaac backend requested, map ACTIVE URIs to USD files under DEMO_ASSET_PATH
    if args.sim_backend == "isaac":
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
        table_x, table_y, table_z = spawn_table_and_get_pose(node, spawn_entity_client, get_entity_state_client)
        if table_x is None:
            return

        # Spawn cubes around the table
        spawn_cubes_around_table(node, spawn_entity_client, table_x, table_y, table_z)
        time.sleep(0.5)
        
        # Spawn Dingo robot (returns cmd_vel publisher or None)
        dingo_cmd_vel_pub = spawn_dingo(node, spawn_entity_client)
        time.sleep(0.5)

        # Spawn UR10 robot
        spawn_ur10(node, spawn_entity_client, table_x, table_y, table_z)
        time.sleep(2.0)

        # Move box and set entity state
        move_cubes_and_step_sim(node, set_state_client, set_entity_state_client, step_sim_client, table_x, table_y, table_z)

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
