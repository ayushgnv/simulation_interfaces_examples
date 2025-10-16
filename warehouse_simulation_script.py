import rclpy
from simulation_interfaces.srv import GetEntityState, SetEntityState, SpawnEntity, DeleteEntity, SetSimulationState, ResetSimulation, GetEntityInfo, LoadWorld, UnloadWorld
from simulation_interfaces.msg import Result, SimulationState
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import os
import numpy as np
import time

DEMO_ASSET_PATH = os.getenv('DEMO_ASSET_PATH')


def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) to a quaternion (w, x, y, z)."""
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    half_yaw = yaw / 2.0
    q = np.array([
        np.cos(half_yaw),  # w
        0.0,               # x
        0.0,               # y  
        np.sin(half_yaw)   # z
    ])
    q = q / np.linalg.norm(q)
    return tuple(q)

def main():
    rclpy.init()
    
    # Initialize main ROS node
    node = rclpy.create_node("warehouse_simulation")
    
    # Create service clients
    set_state_client = node.create_client(SetSimulationState, 'set_simulation_state')
    reset_client = node.create_client(ResetSimulation, 'reset_simulation')
    load_world_client = node.create_client(LoadWorld, 'load_world')
    unload_world_client = node.create_client(UnloadWorld, 'unload_world')
    spawn_entity_client = node.create_client(SpawnEntity, 'spawn_entity')
    get_entity_state_client = node.create_client(GetEntityState, 'get_entity_state')
    set_entity_state_client = node.create_client(SetEntityState, 'set_entity_state')
    get_entity_info_client = node.create_client(GetEntityInfo, 'get_entity_info')
    
    # Create publishers for robot control
    dingo_cmd_vel_pub = None
    ur10_joint_pub = None

    try:
        # Wait for services
        print("Waiting for simulation services...")
        while not set_state_client.wait_for_service(timeout_sec=1.0):
            print("set_simulation_state service not available, waiting...")
        while not load_world_client.wait_for_service(timeout_sec=1.0):
            print("load_world service not available, waiting...")
        while not spawn_entity_client.wait_for_service(timeout_sec=1.0):
            print("spawn_entity service not available, waiting...")
        
        print("All services ready!")

        # Load warehouse world
        print("Loading warehouse world...")
        req = LoadWorld.Request()
        req.uri = os.path.join(DEMO_ASSET_PATH, "Collected_warehouse_with_forklifts/warehouse_with_forklifts.usd")
        future = load_world_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("Warehouse world loaded successfully")
        else:
            print("Failed to load warehouse world: " + future.result().result.error_message)
            return

        time.sleep(1.0)

        # Load a table prop
        print("Spawning warehouse table...")
        table_req = SpawnEntity.Request()
        table_req.name = "warehouse_table"
        table_req.uri = os.path.join(DEMO_ASSET_PATH, "thor_table/thor_table.usd")
        table_req.allow_renaming = False
        table_req.initial_pose = PoseStamped()
        table_req.initial_pose.header.frame_id = "world"
        table_req.initial_pose.pose.position.x = float(-1.0)
        table_req.initial_pose.pose.position.y = float(-1.5)
        table_req.initial_pose.pose.position.z = float(1.19)
        quat = yaw_to_quaternion(1.5708)    
        table_req.initial_pose.pose.orientation.w = float(quat[0])
        table_req.initial_pose.pose.orientation.x = float(quat[1])
        table_req.initial_pose.pose.orientation.y = float(quat[2])
        table_req.initial_pose.pose.orientation.z = float(quat[3])
        future = spawn_entity_client.call_async(table_req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("Table spawned successfully")
        else:
            print("Failed to spawn table: " + future.result().result.error_message)

        time.sleep(0.5)

        # Get table state to make all other positions relative to it
        print("Getting table state for relative positioning...")
        req = GetEntityState.Request()
        req.entity = "warehouse_table"
        future = get_entity_state_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            table_state = future.result().state
            table_x = table_state.pose.position.x
            table_y = table_state.pose.position.y
            table_z = table_state.pose.position.z
            print(f"Table found at position: ({table_x:.2f}, {table_y:.2f}, {table_z:.2f})")
        else:
            print("Failed to get table state - cannot proceed with relative positioning")
            print("Error: " + future.result().result.error_message)
            return

        # Load some cubes around the table top (relative to table position) - mix of red and blue
        print("Spawning cubes around table...")
        cube_configs = [
            # (position, color)
            ((table_x + 0.2, table_y + 0.2, table_z + 0.1), "blue"),   # Top right - blue
            ((table_x - 0.2, table_y + 0.2, table_z + 0.1), "red"),    # Top left - red
            ((table_x + 0.2, table_y - 0.2, table_z + 0.1), "red"),    # Bottom right - red
            ((table_x - 0.2, table_y - 0.2, table_z + 0.1), "blue"),   # Bottom left - blue
            ((table_x, table_y, table_z + 0.1), "blue"),               # Center - blue
            ((table_x + 0.1, table_y, table_z + 0.1), "red"),          # Center right - red
            ((table_x - 0.1, table_y, table_z + 0.1), "red"),          # Center left - red
        ]
        
        for i, (pos, color) in enumerate(cube_configs):
            cube_req = SpawnEntity.Request()
            cube_req.name = f"{color}_cube"
            
            # Choose asset based on color
            if color == "blue":
                cube_req.uri = os.path.join(DEMO_ASSET_PATH, "Collected_blue_block/blue_block.usd")
            else:  # red
                cube_req.uri = os.path.join(DEMO_ASSET_PATH, "Collected_red_block/red_block.usd")

            cube_req.allow_renaming = True
            cube_req.initial_pose = PoseStamped()
            cube_req.initial_pose.header.frame_id = "world"
            cube_req.initial_pose.pose.position.x = float(pos[0])
            cube_req.initial_pose.pose.position.y = float(pos[1])
            cube_req.initial_pose.pose.position.z = float(pos[2])
            cube_req.initial_pose.pose.orientation.w = float(1.0)
            cube_req.initial_pose.pose.orientation.x = float(0.0)
            cube_req.initial_pose.pose.orientation.y = float(0.0)
            cube_req.initial_pose.pose.orientation.z = float(0.0)
            future = spawn_entity_client.call_async(cube_req)
            rclpy.spin_until_future_complete(node, future)
            
            if future.result() and future.result().result.result == Result.RESULT_OK:
                print(f"{color.capitalize()} cube spawned successfully at {pos}")
            else:
                print(f"Failed to spawn {color} cube: " + future.result().result.error_message)

        time.sleep(0.5)

        # Load Dingo robot
        print("Spawning Dingo robot...")
        dingo_req = SpawnEntity.Request()
        dingo_req.name = "dingo_robot"
        dingo_req.uri = os.path.join(DEMO_ASSET_PATH, "dingo/dingo_ROS.usd")  # Assuming dingo asset exists
        dingo_req.entity_namespace = "dingo"
        dingo_req.allow_renaming = False
        dingo_req.initial_pose = PoseStamped()
        dingo_req.initial_pose.header.frame_id = "world"
        dingo_req.initial_pose.pose.position.x = float(-4.0)  # Fixed position
        dingo_req.initial_pose.pose.position.y = float(-3.0)  # Fixed position
        dingo_req.initial_pose.pose.position.z = float(0.0)
        quat = yaw_to_quaternion(0.0)  # Facing forward
        dingo_req.initial_pose.pose.orientation.w = float(quat[0])
        dingo_req.initial_pose.pose.orientation.x = float(quat[1])
        dingo_req.initial_pose.pose.orientation.y = float(quat[2])
        dingo_req.initial_pose.pose.orientation.z = float(quat[3])
        future = spawn_entity_client.call_async(dingo_req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("Dingo robot spawned successfully")
            # Create cmd_vel publisher for dingo
            dingo_cmd_vel_pub = node.create_publisher(Twist, "/dingo/cmd_vel", 10)
        else:
            print("Failed to spawn Dingo robot")

        time.sleep(0.5)

        # Load UR10 robot
        print("Spawning UR10 robot...")
        ur10_req = SpawnEntity.Request()
        ur10_req.name = "ur10_robot"
        ur10_req.uri = os.path.join(DEMO_ASSET_PATH, "Collected_ur10e_robotiq2f-140_ROS/ur10e_robotiq2f-140_ROS.usd")  # Assuming ur10e asset exists
        ur10_req.entity_namespace = "ur10"
        ur10_req.allow_renaming = False
        ur10_req.initial_pose = PoseStamped()
        ur10_req.initial_pose.header.frame_id = "world"
        ur10_req.initial_pose.pose.position.x = float(table_x)  # Match table x position
        ur10_req.initial_pose.pose.position.y = float(table_y - 0.64)  # Match table y position
        ur10_req.initial_pose.pose.position.z = float(table_z)  # On top of the table
        quat = yaw_to_quaternion(-1.5708)  # 180 rotation
        ur10_req.initial_pose.pose.orientation.w = float(quat[0])   
        ur10_req.initial_pose.pose.orientation.x = float(quat[1])
        ur10_req.initial_pose.pose.orientation.y = float(quat[2])
        ur10_req.initial_pose.pose.orientation.z = float(quat[3])
        future = spawn_entity_client.call_async(ur10_req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("UR10 robot spawned successfully")
            # Create joint state publisher for ur10
            ur10_joint_pub = node.create_publisher(JointState, "/ur10/joint_commands", 10)
        else:
            print("Failed to spawn UR10 robot")

        time.sleep(1.0)

        # Start simulation
        print("Starting simulation...")
        req = SetSimulationState.Request()
        req.state.state = SimulationState.STATE_PLAYING
        future = set_state_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("Simulation started successfully")
        else:
            print("Failed to start simulation: " + future.result().result.error_message)

        time.sleep(2.0)

        # Main simulation loop
        for loop_iteration in range(3):
            print(f"\n=== Loop iteration {loop_iteration + 1} ===")
            
            # Get updated table state (in case it moved)
            print("Getting table state...")
            req = GetEntityState.Request()
            req.entity = "warehouse_table"
            future = get_entity_state_client.call_async(req)
            
            rclpy.spin_until_future_complete(node, future)
            
            if future.result() and future.result().result.result == Result.RESULT_OK:
                table_state = future.result().state
                table_x = table_state.pose.position.x
                table_y = table_state.pose.position.y
                table_z = table_state.pose.position.z
                print(f"Table position: ({table_x:.2f}, {table_y:.2f}, {table_z:.2f})")
            else:
                print("Failed to get table state - ending simulation loop")
                print("Error: " + future.result().result.error_message)
                break
            
            # Spawn cardboard boxes at different positions based on loop iteration
            print(f"Spawning obstacle boxes near table (iteration {loop_iteration + 1})...")
            
            # Different box positions for each iteration
            box_positions_by_iteration = [
                # Iteration 1: Boxes around the south side
                [
                    (table_x - 1.5, table_y - 1.0, 0.0, 0.0),      # Southwest
                    (table_x + 0.5, table_y - 2.0, 0.0, 1.5708),   # South, rotated 90°
                    (table_x - 0.8, table_y - 1.8, 0.0, 0.7854),   # Southeast, rotated 45°
                ],
                # Iteration 2: Boxes around the east and west sides  
                [
                    (table_x - 2.0, table_y + 0.5, 0.0, 1.5708),   # West, rotated 90°
                    (table_x + 1.5, table_y - 0.5, 0.0, 0.0),      # East
                    (table_x + 1.8, table_y + 1.2, 0.0, 0.7854),   # Northeast, rotated 45°
                ],
                # Iteration 3: Boxes around the north side
                [
                    (table_x - 0.8, table_y + 1.8, 0.0, 0.7854),   # North, rotated 45°
                    (table_x + 0.3, table_y + 1.5, 0.0, 1.5708),   # Northeast, rotated 90°
                    (table_x - 1.2, table_y + 0.8, 0.0, 0.0),      # Northwest
                ]
            ]
            
            box_positions = box_positions_by_iteration[loop_iteration]
            
            for i, (box_x, box_y, box_z, box_yaw) in enumerate(box_positions):
                box_req = SpawnEntity.Request()
                box_req.name = "obstacle_box"
                box_req.uri = os.path.join(DEMO_ASSET_PATH, "Collected_warehouse_with_forklifts/Props/SM_CardBoxA_02.usd")
                box_req.allow_renaming = True
                box_req.initial_pose = PoseStamped()
                box_req.initial_pose.header.frame_id = "world"
                box_req.initial_pose.pose.position.x = float(box_x)
                box_req.initial_pose.pose.position.y = float(box_y)
                box_req.initial_pose.pose.position.z = float(box_z)
                # Specific rotation for each box
                quat = yaw_to_quaternion(box_yaw)
                box_req.initial_pose.pose.orientation.w = float(quat[0])
                box_req.initial_pose.pose.orientation.x = float(quat[1])
                box_req.initial_pose.pose.orientation.y = float(quat[2])
                box_req.initial_pose.pose.orientation.z = float(quat[3])
                future = spawn_entity_client.call_async(box_req)
                rclpy.spin_until_future_complete(node, future)
                
                if future.result() and future.result().result.result == Result.RESULT_OK:
                    print(f"Obstacle box {i+1} spawned at ({box_x:.2f}, {box_y:.2f})")
                else:
                    print(f"Failed to spawn obstacle box {i+1}: " + future.result().result.error_message)
            
            time.sleep(0.5)

            # Move the Dingo robot towards the table
            if dingo_cmd_vel_pub:
                print("Moving Dingo robot towards table...")
                
                # Calculate direction to table
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
                    
                    # Calculate movement towards table
                    dx = table_x - dingo_x
                    dy = table_y - dingo_y
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 1.5:  # Move closer if too far
                        # Send velocity commands
                        cmd_vel = Twist()
                        cmd_vel.linear.x = 0.3  # Forward speed
                        cmd_vel.angular.z = np.arctan2(dy, dx) * 0.5  # Turn towards table
                        
                        # Send commands for 2 seconds
                        for _ in range(20):
                            dingo_cmd_vel_pub.publish(cmd_vel)
                            time.sleep(0.1)
                        
                        # Stop the robot
                        cmd_vel.linear.x = 0.0
                        cmd_vel.angular.z = 0.0
                        for _ in range(5):
                            dingo_cmd_vel_pub.publish(cmd_vel)
                            time.sleep(0.1)
                    
                    print(f"Dingo robot moved towards table (distance: {distance:.2f}m)")

            # Move the UR10 by sending manual joint commands (arm only)
            if ur10_joint_pub:
                print("Moving UR10 robot joints...")
                
                joint_cmd = JointState()
                joint_cmd.header = Header()
                joint_cmd.header.stamp = node.get_clock().now().to_msg()
                joint_cmd.name = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                                 "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
                
                # Predefined joint positions for each iteration
                joint_positions_by_iteration = [
                    [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],          # Iteration 1: Home position
                    [0.7854, -1.0472, 0.7854, -0.7854, -1.5708, 0.5236],    # Iteration 2: Reach pose
                    [-0.5236, -2.0944, 2.0944, -2.0944, -1.5708, -0.7854],  # Iteration 3: Pickup pose
                ]
                
                # Use specific joint positions for this iteration
                joint_cmd.position = joint_positions_by_iteration[loop_iteration]
                
                # Send joint commands
                for _ in range(10):
                    ur10_joint_pub.publish(joint_cmd)
                    time.sleep(0.2)
                
                print("UR10 joint positions updated")

            time.sleep(0.1)

            # Move robot to a specific location relative to table
            print("Moving Dingo to a specific location...")
            new_x = table_x - 3.0  # 3 meters away from table
            new_y = table_y - 2.5  # 2.5 meters in front of table
            new_yaw = 1.5708       # Face towards table (90 degrees)
            
            # Use set_entity_state to teleport the robot to new position
            from simulation_interfaces.msg import EntityState
            from geometry_msgs.msg import Vector3
            
            req = SetEntityState.Request()
            req.entity = "dingo_robot"
            
            state = EntityState()
            state.pose = Pose()
            state.pose.position = Point(x=float(new_x), y=float(new_y), z=float(0.0))
            quat = yaw_to_quaternion(new_yaw)
            state.pose.orientation = Quaternion(w=float(quat[0]), x=float(quat[1]), 
                                              y=float(quat[2]), z=float(quat[3]))
            state.twist = Twist()
            state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
            state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
            
            req.state = state
            
            future = set_entity_state_client.call_async(req)
            
            rclpy.spin_until_future_complete(node, future)
            
            if future.result() and future.result().result.result == Result.RESULT_OK:
                print(f"Dingo moved to new position: ({new_x:.2f}, {new_y:.2f}, yaw: {new_yaw:.2f})")
            else:
                print("Failed to move Dingo to new position: " + future.result().result.error_message)

            time.sleep(0.1)  # Wait between iterations

        print("\nSimulation loop completed!")

        # Stop simulation
        print("Stopping simulation...")
        req = SetSimulationState.Request()
        req.state.state = SimulationState.STATE_STOPPED
        future = set_state_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("Simulation stopped successfully")
        else:
            print("Failed to stop simulation: " + future.result().result.error_message)

        time.sleep(0.5)

        # Unload world
        print("Unloading world...")
        req = UnloadWorld.Request()
        future = unload_world_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        if future.result() and future.result().result.result == Result.RESULT_OK:
            print("World unloaded successfully")
        else:
            print("Failed to unload world: " + future.result().result.error_message)

        print("Warehouse simulation completed!")
        
    except KeyboardInterrupt:
        print("\nInterrupted! Stopping simulation...")
        
        # Stop simulation
        req = SetSimulationState.Request()
        req.state.state = SimulationState.STATE_STOPPED
        future = set_state_client.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        
        print("Simulation stopped due to interruption!")
        
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
