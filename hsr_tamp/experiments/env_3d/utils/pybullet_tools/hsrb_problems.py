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
    FLOOR_URDF, TABLE_URDF, \
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

PROBLEMS = [
    holding_problem,
    stacking_problem,
    cleaning_problem,
    cooking_problem,
    cleaning_button_problem,
    cooking_button_problem,
]
