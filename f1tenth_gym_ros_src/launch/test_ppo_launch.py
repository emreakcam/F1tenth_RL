import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld = LaunchDescription()
    
    # 1. Konfigürasyon dosyasını çek
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml'
    )
    config_dict = yaml.safe_load(open(config, 'r'))

    # 2. Map Server Parametreleri
    # Exit code -6 genelde lifecycle manager'ın map_server'ı bulamamasından olur.
    # Bu yüzden node isimlerini netleştiriyoruz.
    map_path = config_dict['bridge']['ros__parameters']['map_path'] + '.yaml'

    # --- Düğümler ---

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')]
    )

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server', # İsmin lifecycle manager ile eşleştiğinden emin ol
        parameters=[{'yaml_filename': map_path},
                    {'use_sim_time': True}]
    )

    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}] # Sadece map_server'ı yönetmesi yeterli
    )

    # EGO CAR: URDF ve State Publisher
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'ego_racecar.xacro')]),
                     'use_sim_time': True}],
        remappings=[('/robot_description', 'ego_robot_description')]
    )

    # OPPONENT CAR: URDF ve State Publisher
    # Eğer xacro dosyan yoksa ego'nunkini kullanabilirsin.
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'ego_racecar.xacro')]),
                     'use_sim_time': True}],
        remappings=[('/robot_description', 'opponent_robot_description')]
    )

    # TEST DÜĞÜMÜ (Senin Test Scriptin)
    rl_test_node = Node(
        package='f1tenth_gym_ros', 
        executable='test_f110', 
        name='test_f110_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Aksiyonları ekle
    ld.add_action(rviz_node)
    ld.add_action(bridge_node)
    ld.add_action(map_server_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(ego_robot_publisher)
    ld.add_action(opp_robot_publisher)
    ld.add_action(rl_test_node)

    return ld