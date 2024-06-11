import os
import random
import numpy as np
from itertools import product

from .hsrb_utils import set_arm_conf, set_group_conf, get_carry_conf, get_other_arm, \
    create_gripper, arm_conf, open_arm, close_arm, HSRB_URDF

from .utils import (
    # Setter
    set_base_values, set_point, set_pose, \
    # Getter
    get_pose, get_bodies, get_box_geometry, get_cylinder_geometry, \
    # Utility
    create_body, create_box, create_virtual_box, create_shape_array, create_virtual_cylinder, create_marker, \
    z_rotation, add_data_path, remove_body, load_model, load_pybullet, load_virtual_model, \
    LockRenderer, HideOutput, \
    # Geometry
    Point, Pose, \
    # URDF
    FLOOR_URDF, TABLE_URDF, BLUE_GEAR_URDF, GREEN_GEAR_URDF, RED_GEAR_URDF, \
    YELLOW_SHAFT_URDF, RED_SHAFT_URDF, GEARBOX_BASE_URDF, \
    # Color
    LIGHT_GREY, TAN, GREY)

class Problem(object):
    def __init__(self, robot, arms=tuple(), movable=tuple(), bodies=tuple(), fixed=tuple(), holes=tuple(),
                 grasp_types=tuple(), surfaces=tuple(), sinks=tuple(), stoves=tuple(), buttons=tuple(),
                 init_placeable=tuple(), init_insertable=tuple(),
                 goal_conf=None, goal_holding=tuple(), goal_on=tuple(),
                 goal_inserted=tuple(), goal_cleaned=tuple(), goal_cooked=tuple(),
                 costs=False, body_names={}, body_types=[], base_limits=None):
        self.robot = robot
        self.arms = arms
        self.movable = movable
        self.grasp_types = grasp_types
        self.surfaces = surfaces
        self.sinks = sinks
        self.stoves = stoves
        self.buttons = buttons
        self.init_placeable = init_placeable
        self.init_insertable = init_insertable
        self.goal_conf = goal_conf
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_inserted = goal_inserted
        self.goal_cleaned = goal_cleaned
        self.goal_cooked = goal_cooked
        self.costs = costs
        self.bodies = bodies
        self.body_names = body_names
        self.body_types = body_types
        self.base_limits = base_limits
        self.holes = holes
        self.fixed = fixed # list(filter(lambda b: b not in all_movable, get_bodies()))
        self.gripper = None

    def get_gripper(self, arm='arm', visual=True):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm, visual=visual)

        return self.gripper

    def remove_gripper(self):
        if self.gripper is not None:
            remove_body(self.gripper)
            self.gripper = None

    def __repr__(self):
        return repr(self.__dict__)

#######################################################

def get_fixed_bodies(problem):
    return problem.fixed

def create_hsr(fixed_base=True, torso=0.0):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/hsrb_description')
    add_data_path(directory)

    hsr_path = HSRB_URDF
    hsr_init_pose = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
    with LockRenderer():
        with HideOutput():
            hsr = load_model(hsr_path, pose=hsr_init_pose, fixed_base=fixed_base)
        set_group_conf(hsr, 'torso', [torso])

    return hsr

def create_floor(**kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    add_data_path(directory)
    return load_pybullet(FLOOR_URDF, **kwargs)

def create_table(width=0.6, length=1.2, height=0.73, thickness=0.03, radius=0.015,
                 top_color=LIGHT_GREY, leg_color=TAN, cylinder=True, **kwargs):
    surface = get_box_geometry(width, length, thickness)
    surface_pose = Pose(Point(z=height - thickness/2.))

    leg_height = height-thickness

    if cylinder:
        leg_geometry = get_cylinder_geometry(radius, leg_height)
    else:
        leg_geometry = get_box_geometry(width=2*radius, length=2*radius, height=leg_height)

    legs = [leg_geometry for _ in range(4)]
    leg_center = np.array([width, length])/2. - radius*np.ones(2)
    leg_xys = [np.multiply(leg_center, np.array(signs))
               for signs in product([-1, +1], repeat=len(leg_center))]
    leg_poses = [Pose(point=[x, y, leg_height/2.]) for x, y in leg_xys]

    geoms = [surface] + legs
    poses = [surface_pose] + leg_poses
    colors = [top_color] + len(legs)*[leg_color]

    collision_id, visual_id = create_shape_array(geoms, poses, colors)
    body = create_body(collision_id, visual_id, **kwargs)

    return body

def create_door():
    return load_pybullet("data/door.urdf")

def create_gear(color='red', **kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    add_data_path(directory)

    if color == 'blue':
        urdf_path = BLUE_GEAR_URDF
    elif color == 'green':
        urdf_path = GREEN_GEAR_URDF
    elif color == 'red':
        urdf_path = RED_GEAR_URDF
    else:
        return None
    return load_pybullet(urdf_path, **kwargs)

def create_shaft(color='red', **kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    add_data_path(directory)

    if color == 'yellow':
        urdf_path = YELLOW_SHAFT_URDF
    elif color == 'red':
        urdf_path = RED_SHAFT_URDF
    else:
        return None
    return load_pybullet(urdf_path, **kwargs)

def create_gearbox_base(**kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    add_data_path(directory)
    return load_pybullet(GEARBOX_BASE_URDF, fixed_base=True, **kwargs)

def create_virtual_gear(color='red', **kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    add_data_path(directory)

    if color == 'blue':
        urdf_path = BLUE_GEAR_URDF
    elif color == 'green':
        urdf_path = GREEN_GEAR_URDF
    elif color == 'red':
        urdf_path = RED_GEAR_URDF
    else:
        return None
    return load_virtual_model(urdf_path, **kwargs)

def create_virtual_shaft(color='red', **kwargs):
    directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    add_data_path(directory)

    if color == 'yellow':
        urdf_path = YELLOW_SHAFT_URDF
    elif color == 'red':
        urdf_path = RED_SHAFT_URDF
    else:
        return None
    return load_virtual_model(urdf_path, **kwargs)

#######################################################

TABLE_MAX_Z = 0.6265

def holding_problem(arm='arm', grasp_type='side'):
    hsr = create_hsr()
    initial_conf = get_carry_conf(arm, grasp_type)

    set_base_values(hsr, (0, 0, 0))
    set_arm_conf(hsr, arm, initial_conf)
    open_arm(hsr, arm)

    plane = create_floor(fixed_base=True)

    box = create_box(.04, .04, .04)
    set_point(box, (1.3, 0.0, 0.22))

    table = create_box(0.65, 1.2, 0.20, color=(1, 1, 1, 1))
    set_point(table, (1.5, 0.0, 0.1))

    block_names = {box: 'block'}

    return Problem(robot=hsr, movable=[box], arms=[arm], body_names=block_names,
                   grasp_types=[grasp_type], surfaces=[table], goal_holding=[(arm, box)])

def stacking_problem(arm='arm', grasp_type='side'):
    hsr = create_hsr()
    initial_conf = get_carry_conf(arm, grasp_type)

    set_base_values(hsr, (0, 0, 0))
    set_arm_conf(hsr, arm, initial_conf)
    open_arm(hsr, arm)

    plane = create_floor(fixed_base=True)

    block1 = create_box(.04, .04, .04, color=(0.0, 1.0, 0.0, 1.0))
    set_point(block1, (1.5, 0.45, 0.275))

    table1 = create_box(0.5, 0.5, 0.25, color=(.25, .25, .75, 1))
    set_point(table1, (1.5, 0.5, 0.125))

    table2 = create_box(0.5, 0.5, 0.25, color=(.75, .25, .25, 1))
    set_point(table2, (1.5, -0.5, 0.125))

    block_names = {block1: 'block', table1: 'table1', table2: 'table2'}

    return Problem(robot=hsr, movable=[block1], arms=[arm], body_names=block_names,
                   grasp_types=[grasp_type], surfaces=[table1, table2],
                   goal_on=[(block1, table2)])

#######################################################

def create_kitchen(w=.5, h=.2):
    plane = create_floor(fixed_base=True)

    table = create_box(w, w, h, color=(.75, .75, .75, 1))
    set_point(table, (2, 0, h/2))

    mass = 1
    cabbage = create_box(.07, .07, .1, mass=mass, color=(0, 1, 0, 1))
    set_point(cabbage, (1.80, 0, h + .1/2))

    sink = create_box(w, w, h, color=(.25, .25, .75, 1))
    set_point(sink, (0, 2, h/2))

    stove = create_box(w, w, h, color=(.75, .25, .25, 1))
    set_point(stove, (0, -2, h/2))

    return table, cabbage, sink, stove

#######################################################

def cleaning_problem(arm='arm', grasp_type='side'):
    initial_conf = get_carry_conf(arm, grasp_type)

    hsr = create_hsr()
    set_arm_conf(hsr, arm, initial_conf)

    table, cabbage, sink, stove = create_kitchen()

    return Problem(robot=hsr, movable=[cabbage], arms=[arm], grasp_types=[grasp_type],
                   surfaces=[table, sink, stove], sinks=[sink], stoves=[stove],
                   goal_cleaned=[cabbage])

def cooking_problem(arm='arm', grasp_type='side'):
    initial_conf = get_carry_conf(arm, grasp_type)

    hsr = create_hsr()
    set_arm_conf(hsr, arm, initial_conf)

    table, cabbage, sink, stove = create_kitchen()

    return Problem(robot=hsr, movable=[cabbage], arms=[arm], grasp_types=[grasp_type],
                   surfaces=[table, sink, stove], sinks=[sink], stoves=[stove],
                   goal_cooked=[cabbage])

def cleaning_button_problem(arm='arm', grasp_type='side'):
    initial_conf = get_carry_conf(arm, grasp_type)

    hsr = create_hsr()
    set_arm_conf(hsr, arm, initial_conf)

    table, cabbage, sink, stove = create_kitchen()

    d = 0.1
    sink_button = create_box(d, d, d, color=(0, 0, 0, 1))
    set_pose(sink_button, ((0, 2-(.5+d)/2, .7-d/2), z_rotation(np.pi/2)))

    stove_button = create_box(d, d, d, color=(0, 0, 0, 1))
    set_pose(stove_button, ((0, -2+(.5+d)/2, .7-d/2), z_rotation(-np.pi/2)))

    return Problem(robot=hsr, movable=[cabbage], arms=[arm], grasp_types=[grasp_type],
                   surfaces=[table, sink, stove], sinks=[sink], stoves=[stove],
                   buttons=[(sink_button, sink), (stove_button, stove)],
                   goal_conf=get_pose(hsr), goal_holding=[(arm, cabbage)], goal_cleaned=[cabbage])

def cooking_button_problem(arm='arm', grasp_type='side'):
    initial_conf = get_carry_conf(arm, grasp_type)

    hsr = create_hsr()
    set_arm_conf(hsr, arm, initial_conf)

    table, cabbage, sink, stove = create_kitchen()

    d = 0.1
    sink_button = create_box(d, d, d, color=(0, 0, 0, 1))
    set_pose(sink_button, ((0, 2-(.5+d)/2, .7-d/2), z_rotation(np.pi/2)))

    stove_button = create_box(d, d, d, color=(0, 0, 0, 1))
    set_pose(stove_button, ((0, -2+(.5+d)/2, .7-d/2), z_rotation(-np.pi/2)))

    return Problem(robot=hsr, movable=[cabbage], arms=[arm], grasp_types=[grasp_type],
                   surfaces=[table, sink, stove], sinks=[sink], stoves=[stove],
                   buttons=[(sink_button, sink), (stove_button, stove)],
                   goal_conf=get_pose(hsr), goal_holding=[(arm, cabbage)], goal_cooked=[cabbage])

def gearbox_problem(arm='arm', grasp_type='side', randomize=True):
    plane = create_floor(fixed_base=True)

    table_width = 0.47 # 47cm (measured)
    table_depth = 1.82 # 182cm (measured)
    table_height = 0.11 # 11cm (measured)
    table_x_pos = 1.2 # 120cm (measured)
    table_y_pos = -0.2 # -20cm (measured)
    table_z_pos = 0.055 # 11/2cm (measured)

    gear_base_width = 0.03 # 3cm (measured)
    gear_base_depth = 0.08 # 8cm (measured)
    gear_base_height = 0.06 # 6cm (measured)

    base_width = 0.27 # 27cm (measured)
    base_depth = 0.27 # 27cm (measured)
    base_height = 0.01 # 1cm (measured)

    green_gear_xy = np.array([table_x_pos, 0.30])
    blue_gear_xy = np.array([table_x_pos, -0.50])
    red_gear_xy = np.array([table_x_pos, -0.75])
    red_shaft_xy = np.array([table_x_pos, 0.50])
    yellow_shaft_xy = np.array([table_x_pos, -0.30])
    gearbox_base_xy = np.array([table_x_pos-0.10, 0.0])

    vleft_hole_xy = np.array([table_x_pos-0.003, 0.0975])
    vright_hole_xy = np.array([table_x_pos-0.003, -0.0965])
    vleft_shaft_xy = np.array([table_x_pos-0.003, 0.0975])
    vright_shaft_xy = np.array([table_x_pos-0.003, -0.0965])
    vmiddle_shaft_xy = np.array([table_x_pos-0.10, 0.0])

    # Domain randomization
    if randomize:
        # Randomize xy position for gearbox parts
        rand_x_pos = np.random.uniform(low=-0.05, high=0.05, size=1)
        rand_y_pos = np.random.uniform(low=-0.05, high=0.05, size=1)
        rand_xy_pos = np.concatenate((rand_x_pos, rand_y_pos))
        green_gear_xy += rand_xy_pos
        blue_gear_xy += rand_xy_pos
        red_gear_xy += rand_xy_pos
        red_shaft_xy += rand_xy_pos
        yellow_shaft_xy += rand_xy_pos

        # Randomize xy position for gearbox_base
        rand_x_pos = np.random.uniform(low=-0.05, high=0.05, size=1)
        rand_y_pos= np.random.uniform(low=-0.05, high=0.05, size=1)
        gearbox_base_rand = np.concatenate((rand_x_pos, rand_y_pos))
        gearbox_base_xy += gearbox_base_rand
        vleft_hole_xy += gearbox_base_rand
        vright_hole_xy += gearbox_base_rand
        vleft_shaft_xy += gearbox_base_rand
        vright_shaft_xy += gearbox_base_rand
        vmiddle_shaft_xy += gearbox_base_rand

        table_x_pos += rand_x_pos
        table_y_pos += rand_y_pos

    # Create table
    table = create_box(table_width, table_depth, table_height, color=(1, 0.9, 0.9, 1))
    set_point(table, (table_x_pos, table_y_pos, table_z_pos))

    # Create hsr
    hsr = create_hsr()
    initial_conf = get_carry_conf(arm, grasp_type)

    # Set hsr position
    set_base_values(hsr, (0, 0, 0))
    set_arm_conf(hsr, arm, initial_conf)
    open_arm(hsr, arm)

    # Create gearbox environment
    green_gear = create_gear(color='green')
    set_point(green_gear, (*green_gear_xy, table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    blue_gear = create_gear(color='blue')
    set_point(blue_gear, (*blue_gear_xy, table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    red_gear = create_gear(color='red')
    set_point(red_gear, (*red_gear_xy, table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    red_shaft = create_shaft(color='red')
    red_shaft_height = 0.115 # 11.5cm (measured)
    set_point(red_shaft, (*red_shaft_xy, table_height+red_shaft_height), (0.0, 0.0, 0.0, 1.0))

    yellow_shaft = create_shaft(color='yellow')
    yellow_shaft_height = 0.115 # 11.5cm (measured)
    set_point(yellow_shaft, (*yellow_shaft_xy, table_height+yellow_shaft_height), (0.0, 0.0, 0.0, 1.0))

    gearbox_base = create_gearbox_base()
    set_point(gearbox_base, (*gearbox_base_xy, table_height+base_height), (0.0, 0.0, 0.0, 1.0))

    # Create virtual base for gear
    box_1 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_1, (*green_gear_xy, table_height+gear_base_height/2))

    box_2 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_2, (*blue_gear_xy, table_height+gear_base_height/2))

    box_3 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_3, (*red_gear_xy, table_height+gear_base_height/2))

    box_4 = create_box(base_width, base_depth, base_height, color=GREY) # For gear_base
    set_point(box_4, (gearbox_base_xy[0]+0.11, gearbox_base_xy[1], table_height+base_height/2), (0., 0., 0.38268343, 0.92387953))

    # Create virtual target surfaces
    virtual_left_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(virtual_left_hole, (*vleft_hole_xy, table_height+0.145), (0.0, 0.0, 0.0, 1.0))

    virtual_right_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(virtual_right_hole, (*vright_hole_xy, table_height+0.145), (0.0, 0.0, 0.0, 1.0))

    virtual_left_shaft = create_virtual_cylinder(radius=0.015, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(virtual_left_shaft, (*vleft_shaft_xy, table_height+0.190), (0.0, 0.0, 0.0, 1.0))

    virtual_right_shaft = create_virtual_cylinder(radius=0.015, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(virtual_right_shaft, (*vright_shaft_xy, table_height+0.200), (0.0, 0.0, 0.0, 1.0))

    virtual_middle_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(virtual_middle_shaft, (*vmiddle_shaft_xy, table_height+0.195), (0.0, 0.0, 0.0, 1.0))

    # Create virtual target holes
    left_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(left_hole, (*vleft_shaft_xy, table_height+0.055), (0.0, 0.0, 0.0, 1.0))

    right_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(right_hole, (*vright_shaft_xy, table_height+0.055), (0.0, 0.0, 0.0, 1.0))

    left_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(left_shaft, (*vleft_hole_xy, table_height+0.100), (0.0, 0.0, 0.0, 1.0))

    right_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(right_shaft, (*vright_hole_xy, table_height+0.110), (0.0, 0.0, 0.0, 1.0))

    middle_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(middle_shaft, (*vmiddle_shaft_xy, table_height+0.105), (0.0, 0.0, 0.0, 1.0))

    gearbox_names = {green_gear: 'gear1', blue_gear: 'gear2', red_gear: 'gear3',
                     red_shaft: 'shaft1', yellow_shaft: 'shaft2', gearbox_base: 'gearbox_base',
                     box_1: 'box_1', box_2: 'box_2', box_3: 'box_3', box_4: 'box_4', table: 'table'}

    return Problem(robot=hsr,
                   body_names=gearbox_names,
                   arms=[arm],
                   grasp_types=[grasp_type],
                   movable=[green_gear, blue_gear, red_gear, yellow_shaft, red_shaft],
                   bodies=[gearbox_base, box_1, box_2, box_3, box_4, table, green_gear, blue_gear, red_gear,
                           yellow_shaft, red_shaft, gearbox_base],
                   fixed=[table, gearbox_base, box_1, box_2, box_3, box_4],
                   surfaces=[virtual_left_hole, virtual_right_hole, virtual_left_shaft, virtual_right_shaft, virtual_middle_shaft],
                   holes=[left_shaft, right_shaft, left_hole, right_hole, middle_shaft],
                   init_placeable=[
                        (red_shaft, virtual_left_hole), (yellow_shaft, virtual_right_hole),
                        (green_gear, virtual_left_shaft), (blue_gear, virtual_right_shaft),
                        (red_gear, virtual_middle_shaft)],
                   init_insertable=[
                        (red_shaft, left_hole), (yellow_shaft, right_hole),
                        (green_gear, left_shaft), (blue_gear, right_shaft),
                        (red_gear, middle_shaft)],
                   goal_inserted=[
                        (red_shaft, left_hole), (yellow_shaft, right_hole),
                        (green_gear, red_shaft), (blue_gear, yellow_shaft),
                        (red_gear, middle_shaft)])

def real_gearbox_problem(observations, arm='arm', grasp_type='side'):
    robot_poses, object_poses = observations

    def parse_observation(object_name, object_poses):
        object_pose = object_poses[object_name]
        rigid_pose = ((object_pose[0][0],
                       object_pose[0][1],
                       object_pose[0][2]),
                      (object_pose[1][0],
                       object_pose[1][1],
                       object_pose[1][2],
                       object_pose[1][3]))
        return rigid_pose

    plane = create_floor(fixed_base=True)

    table_width = 0.47 # 47cm (measured)
    table_depth = 1.82 # 182cm (measured)
    table_height = 0.11 # 11cm (measured)
    table_x_pos = 1.2 # 120cm (measured)
    table_y_pos = -0.2 # -20cm (measured)
    table_z_pos = 0.055 # 11/2cm (measured)

    table = create_box(table_width, table_depth, table_height, color=(1, 0.9, 0.9, 1))
    set_point(table, (table_x_pos, table_y_pos, table_z_pos))

    hsr = create_hsr()

    set_base_values(hsr, robot_poses[:3]) # odom_x, odom_y, odom_rz
    set_arm_conf(hsr, arm, robot_poses[3:]) # 5 dimensions
    open_arm(hsr, arm)

    gear_base_width = 0.03 # 3cm (measured)
    gear_base_depth = 0.08 # 8cm (measured)
    gear_base_height = 0.06 # 6cm (measured)

    base_width = 0.267 # 26.7cm (measured)
    base_depth = 0.267 # 26.7cm (measured)
    base_height = 0.01 # 1cm (measured)

    # Create gearbox environment
    green_gear = create_gear(color='green')
    green_gear_position, _ = parse_observation('gear1', object_poses) # 3 dim
    set_point(green_gear, (green_gear_position[0], green_gear_position[1], table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    blue_gear = create_gear(color='blue')
    blue_gear_position, _ = parse_observation('gear2', object_poses) # 3 dim
    set_point(blue_gear, (blue_gear_position[0], blue_gear_position[1], table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    red_gear = create_gear(color='red')
    red_gear_position, _ = parse_observation('gear3', object_poses)
    set_point(red_gear, (red_gear_position[0], red_gear_position[1], table_height+gear_base_height), (0.0, 0.0, 0.0, 1.0))

    red_shaft = create_shaft(color='red')
    red_shaft_position, _ = parse_observation('shaft1', object_poses)
    red_shaft_height = 0.115 # 11.5cm (measured)
    set_point(red_shaft, (red_shaft_position[0], red_shaft_position[1], table_height+red_shaft_height), (0.0, 0.0, 0.0, 1.0))

    yellow_shaft = create_shaft(color='yellow')
    yellow_shaft_position, _ = parse_observation('shaft2', object_poses)
    yellow_shaft_height = 0.115 # 11.5cm (measured)
    set_point(yellow_shaft, (yellow_shaft_position[0], yellow_shaft_position[1], table_height+yellow_shaft_height), (0.0, 0.0, 0.0, 1.0))

    gearbox_base = create_gearbox_base()
    gearbox_base_position, _ = parse_observation('base', object_poses)
    gearbox_base_offset = 0.265 # 27.0cm (measured)
    gearbox_x_pos = gearbox_base_position[0] - gearbox_base_offset
    gearbox_y_pos = gearbox_base_position[1]
    set_point(gearbox_base, (gearbox_x_pos, gearbox_y_pos, table_height+base_height), (0.0, 0.0, 0.0, 1.0))

    # Create base for gear
    box_1 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_1, (green_gear_position[0], green_gear_position[1], table_height+gear_base_height/2))

    box_2 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_2, (blue_gear_position[0], blue_gear_position[1], table_height+gear_base_height/2))

    box_3 = create_box(gear_base_width, gear_base_depth, gear_base_height, color=GREY)
    set_point(box_3, (red_gear_position[0], red_gear_position[1], table_height+gear_base_height/2))

    box_4 = create_box(base_width, base_depth, base_height, color=GREY)
    set_point(box_4, (gearbox_base_position[0]-0.15, gearbox_base_position[1], table_height+base_height/2), (0., 0., 0.38268343, 0.92387953))

    # Create virtual target surfaces for place
    virtual_left_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    left_offset = (0.0975, 0.0975) # horizontal 9.75cm, vertical 9.75cm (measured)
    left_hole_position = (gearbox_x_pos+left_offset[0], gearbox_y_pos+left_offset[1], table_height+0.145) # (p-offset+0.015, 0.115, h+.15/2+0.0225)
    set_point(virtual_left_hole, left_hole_position, (0.0, 0.0, 0.0, 1.0))

    virtual_right_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    right_offset = (0.0965, -0.0965) # horizontal 9.65cm, vertical 9.65cm (measured)
    right_hole_position = (gearbox_x_pos+right_offset[0], gearbox_y_pos+right_offset[1], table_height+0.145) # (p-offset+0.015, -0.015, h+.165/2+0.015)
    set_point(virtual_right_hole, right_hole_position, (0.0, 0.0, 0.0, 1.0))

    virtual_left_shaft = create_virtual_cylinder(radius=0.015, height=0.01, specular=(0.0, 0.0, 0.0))
    vleft_shaft_position = (gearbox_x_pos+left_offset[0], gearbox_y_pos+left_offset[1], table_height+0.190) # (p-offset+0.015, 0.115, h+.055/2+0.0175)
    set_point(virtual_left_shaft, vleft_shaft_position, (0.0, 0.0, 0.0, 1.0))

    virtual_right_shaft = create_virtual_cylinder(radius=0.015, height=0.01, specular=(0.0, 0.0, 0.0))
    vright_shaft_position = (gearbox_x_pos+right_offset[0], gearbox_y_pos+right_offset[1], table_height+0.200) # (p-offset+0.015, -0.015, h+.075/2+0.02)
    set_point(virtual_right_shaft, vright_shaft_position, (0.0, 0.0, 0.0, 1.0))

    virtual_middle_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    vmiddle_shaft_position = (gearbox_x_pos, gearbox_y_pos, table_height+0.195) # (p-offset-0.05, 0.05, h+.025+0.02)
    set_point(virtual_middle_shaft, vmiddle_shaft_position, (0.0, 0.0, 0.0, 1.0))

    # Create virtual target holes for insert
    left_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(left_hole, (left_hole_position[0], left_hole_position[1], table_height+0.085), (0.0, 0.0, 0.0, 1.0))

    right_hole = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(right_hole, (right_hole_position[0], right_hole_position[1], table_height+0.10), (0.0, 0.0, 0.0, 1.0))

    left_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(left_shaft, (vleft_shaft_position[0], vleft_shaft_position[1], table_height+0.145), (0.0, 0.0, 0.0, 1.0))

    right_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(right_shaft, (vright_shaft_position[0], vright_shaft_position[1], table_height+0.165), (0.0, 0.0, 0.0, 1.0))

    middle_shaft = create_virtual_cylinder(radius=0.005, height=0.01, specular=(0.0, 0.0, 0.0))
    set_point(middle_shaft, (vmiddle_shaft_position[0], vmiddle_shaft_position[1], table_height+0.145), (0.0, 0.0, 0.0, 1.0))

    gearbox_names = {green_gear: 'gear1', blue_gear: 'gear2', red_gear: 'gear3',
                     red_shaft: 'shaft1', yellow_shaft: 'shaft2', gearbox_base: 'gearbox_base',
                     box_1: 'box_1', box_2: 'box_2', box_3: 'box_3', box_4: 'box_4', table: 'table'}

    return Problem(robot=hsr,
                   body_names=gearbox_names,
                   arms=[arm],
                   grasp_types=[grasp_type],
                   movable=[green_gear, blue_gear, red_gear, yellow_shaft, red_shaft],
                   fixed=[table, gearbox_base, box_1, box_2, box_3, box_4],
                   bodies=[gearbox_base, box_1, box_2, box_3, box_4, table, green_gear, blue_gear, red_gear,
                           yellow_shaft, red_shaft, gearbox_base],
                   surfaces=[virtual_left_hole, virtual_right_hole, virtual_left_shaft, virtual_right_shaft, virtual_middle_shaft],
                   holes=[left_shaft, right_shaft, left_hole, right_hole, middle_shaft],
                   init_placeable=[
                        (red_shaft, virtual_left_hole), (yellow_shaft, virtual_right_hole),
                        (green_gear, virtual_left_shaft), (blue_gear, virtual_right_shaft),
                        (red_gear, virtual_middle_shaft)],
                   init_insertable=[
                        (red_shaft, left_hole), (yellow_shaft, right_hole),
                        (green_gear, left_shaft), (blue_gear, right_shaft),
                        (red_gear, middle_shaft)],
                   goal_inserted=[
                        (red_shaft, left_hole), (yellow_shaft, right_hole),
                        (green_gear, red_shaft), (blue_gear, yellow_shaft),
                        (red_gear, middle_shaft)])

def assembly_problem(w=0.47, d=1.5, h=0.105, p=1.82, offset=0.08, arm='arm', grasp_type='side'):
    def get_stable_gen(body, surface, fixed=[]):
        from .hsrb_primitives import sample_placement, pairwise_collision
        while True:
            pose = sample_placement(body, surface)
            if (pose is None) or any(pairwise_collision(body, b) for b in fixed):
                if not pose is None:
                    pass
                continue
            return pose

    # Create plane
    plane = create_floor()

    # Create tables
    table1 = create_box(w, d, h, color=(1, 0.9, 0.9, 1))
    set_point(table1, (p, -0.1, h/2))

    table2 = create_box(w, d, h, color=(1, 0.9, 0.9, 1))
    set_point(table2, (-p, -0.1, h/2))

    table3 = create_box(d, w, h, color=(1, 0.9, 0.9, 1))
    set_point(table3, (0.0, 2.0, h/2))

    table4 = create_box(d, w, h, color=(1, 0.9, 0.9, 1))
    set_point(table4, (0.0, -2.0, h/2))

    tables = [table1, table2, table3, table4]

    # Create hsr
    hsr = create_hsr()
    initial_conf = get_carry_conf(arm, grasp_type)

    set_base_values(hsr, (0, 0, 0))
    set_arm_conf(hsr, arm, initial_conf)
    open_arm(hsr, arm)

    # Create gearbox environment
    h_offset = 0.05
    num_gear = 3
    num_shaft = 2
    gear_color_list = ['green', 'blue', 'red']
    shaft_color_list = ['red', 'yellow']

    movable_list = []
    for i in range(num_gear):
        # Create gear
        gear = create_gear(color=gear_color_list[i])

        # Sample random position for gear
        surface = np.random.choice(tables, 1)[0]
        sampled_pose = get_stable_gen(gear, surface)

        x, y = sampled_pose[0][0], sampled_pose[0][1]

        set_point(gear, (x, y, h+h_offset))

        # Create virtual base for gear
        box = create_box(0.02, 0.06, h_offset, color=GREY)
        set_point(box, (x, y, h+h_offset/2))

        movable_list.append(gear)

    for j in range(num_shaft):
        # Create shaft
        shaft = create_shaft(color=shaft_color_list[j])

        # Sample random position for shaft
        surface = np.random.choice(tables, 1)[0]
        sampled_pose = get_stable_gen(shaft, surface)

        x, y = sampled_pose[0][0], sampled_pose[0][1]

        set_point(shaft, (x, y, h+.15/2))

        movable_list.append(shaft)

    # Create gearbox base
    gearbox_base = create_gearbox_base()
    set_point(gearbox_base, (p-offset-0.05, 0.05, h), (0.0, 0.0, 0.0, 1.0))

    # Create virtual target gearbox environment
    virtual_left_hole = create_virtual_cylinder(radius=0.005, height=0.01)
    set_point(virtual_left_hole, (p-offset+0.015, 0.115, h+.15/2), (0.0, 0.0, 0.0, 1.0))

    virtual_right_hole = create_virtual_cylinder(radius=0.005, height=0.01)
    set_point(virtual_right_hole, (p-offset+0.015, -0.015, h+.165/2), (0.0, 0.0, 0.0, 1.0))

    virtual_left_shaft = create_virtual_cylinder(radius=0.015, height=0.01)
    set_point(virtual_left_shaft, (p-offset+0.015, 0.115, h+.055/2), (0.0, 0.0, 0.0, 1.0))

    virtual_right_shaft = create_virtual_cylinder(radius=0.015, height=0.01)
    set_point(virtual_right_shaft, (p-offset+0.015, -0.015, h+.075/2), (0.0, 0.0, 0.0, 1.0))

    virtual_middle_shaft = create_virtual_cylinder(radius=0.005, height=0.01)
    set_point(virtual_middle_shaft, (p-offset-0.05, 0.05, h+.025), (0.0, 0.0, 0.0, 1.0))

    return Problem(robot=hsr, movable=movable_list, arms=[arm], grasp_types=[grasp_type],
                   surfaces=[virtual_left_hole, virtual_right_hole, virtual_left_shaft, virtual_right_shaft, virtual_middle_shaft],
                   goal_on=[(movable_list[0], virtual_left_hole), (movable_list[1], virtual_right_hole),
                            (movable_list[2], virtual_left_shaft), (movable_list[3], virtual_right_shaft),
                            (movable_list[4], virtual_middle_shaft)])

PROBLEMS = [
    holding_problem,
    stacking_problem,
    cleaning_problem,
    cooking_problem,
    cleaning_button_problem,
    cooking_button_problem,
    gearbox_problem,
    assembly_problem,
    real_gearbox_problem
]
