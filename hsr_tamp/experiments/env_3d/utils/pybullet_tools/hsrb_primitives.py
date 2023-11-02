from __future__ import print_function

import os
import copy
import json
import time
import random
import numpy as np
import pybullet as p

from itertools import islice, count
from scipy.spatial.transform import Rotation as R

from .ikfast.hsrb.ik import is_ik_compiled, hsr_inverse_kinematics
from .hsrb_utils import (
    # Getter
    get_gripper_joints, get_carry_conf, get_top_grasps, get_side_grasps, get_x_presses, \
    get_group_joints, get_gripper_link, get_arm_joints, get_group_conf, get_base_joints, \
    get_base_arm_joints, get_torso_joints, get_base_torso_arm_joints, \
    # Utility
    open_arm, arm_conf, base_arm_conf, learned_pose_generator, joints_from_names, compute_grasp_width, \
    # Constant
    HSR_TOOL_FRAMES, HSR_GROUPS, HSR_GRIPPER_ROOTS, TOP_HOLDING_ARM, SIDE_HOLDING_ARM, GET_GRASPS)
from .utils import (
    # Getter
    get_joint_positions, get_distance, get_min_limit, get_relative_pose, get_aabb, get_unit_vector, \
    get_moving_links, get_custom_limits, get_custom_limits_with_base, get_body_name, get_bodies, get_extend_fn, \
    get_name, get_link_pose, get_point, get_pose, \
    # Setter
    set_pose, is_placement, set_joint_positions, is_inserted, \
    # Utility
    invert, multiply, all_between, pairwise_collision, sample_placement, sample_insertion, \
    waypoints_from_path, unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, \
    pose_from_base_values, uniform_pose_generator, add_fixed_constraint, remove_debug, \
    remove_fixed_constraint, disable_real_time, enable_real_time, enable_gravity, step_simulation, \
    joint_controller_hold, add_segments, link_from_name, interpolate_poses, plan_linear_motion, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, wait_if_gui, flatten_links, create_marker, \
    BodySaver, Attachment, \
    # Constant
    BASE_LINK)

BASE_EXTENT = 3.5
BASE_LIMITS = (-BASE_EXTENT*np.ones(2), BASE_EXTENT*np.ones(2))
GRASP_LENGTH = 0.03
APPROACH_DISTANCE = 0.1 + GRASP_LENGTH
SELF_COLLISIONS = False

##################################################

def get_base_limits():
    return BASE_LIMITS

##################################################

class Pose(object):
    num = count()
    def __init__(self, body, value=None, support=None, extra=None, init=False):
        self.body = body
        if value is None:
            value = get_pose(self.body)
        self.value = tuple(value)
        self.support = support
        self.extra= extra
        self.init = init
        self.index = next(self.num)

    @property
    def bodies(self):
        return flatten_links(self.body)

    def assign(self):
        set_pose(self.body, self.value)

    def iterate(self):
        yield self

    def to_base_conf(self):
        values = base_values_from_pose(self.value)
        return Conf(self.body, range(len(values)), values)

    def __repr__(self):
        index = self.index
        return 'p{}'.format(index)

class Grasp(object):
    def __init__(self, grasp_type, body, value, approach, carry):
        self.grasp_type = grasp_type
        self.body = body
        self.value = tuple(value)
        self.approach = tuple(approach)
        self.carry = tuple(carry)

    def get_attachment(self, robot, arm):
        tool_link = link_from_name(robot, HSR_TOOL_FRAMES[arm])
        return Attachment(robot, tool_link, self.value, self.body)

    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)

class Conf(object):
    def __init__(self, body, joints, values=None, init=False):
        self.body = body
        self.joints = joints
        if values is None:
            values = get_joint_positions(self.body, self.joints)
        self.values = tuple(values)
        self.init = init

    @property
    def bodies(self):
        return flatten_links(self.body, get_moving_links(self.body, self.joints))

    def assign(self):
        set_joint_positions(self.body, self.joints, self.values)

    def iterate(self):
        yield self

    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)
    
class State(object):
    def __init__(self, attachments={}, cleaned=set(), cooked=set()):
        self.poses = {body: Pose(body, get_pose(body))
                      for body in get_bodies() if body not in attachments}
        self.grasps = {}
        self.attachments = attachments
        self.cleaned = cleaned
        self.cooked = cooked

    def assign(self):
        for attachment in self.attachments.values():
            attachment.assign()

#####################################

class Command(object):
    def control(self, dt=0):
        raise NotImplementedError()

    def apply(self, state, **kwargs):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

class Commands(object):
    def __init__(self, state, savers=[], commands=[]):
        self.state = state
        self.savers = tuple(savers)
        self.commands = tuple(commands)

    def assign(self):
        for saver in self.savers:
            saver.restore()
        return copy.copy(self.state)

    def apply(self, state, **kwargs):
        for command in self.commands:
            for result in command.apply(state, **kwargs):
                yield result

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)

#####################################

class Trajectory(Command):
    _draw = False
    def __init__(self, path):
        self.path = tuple(path)

    def apply(self, state, sample=1):
        handles = add_segments(self.to_points()) if self._draw and has_gui() else []
        for conf in self.path[::sample]:
            conf.assign()
            yield
        end_conf = self.path[-1]
        if isinstance(end_conf, Pose):
            state.poses[end_conf.body] = end_conf
        for handle in handles:
            remove_debug(handle)

    def control(self, dt=0, **kwargs):
        for conf in self.path:
            if isinstance(conf, Pose):
                conf = conf.to_base_conf()
            for _ in joint_controller_hold(conf.body, conf.joints, conf.values):
                step_simulation()
                time.sleep(dt)

    def to_points(self, link=BASE_LINK):
        points = []
        for conf in self.path:
            with BodySaver(conf.body):
                conf.assign()
                point = np.array(get_group_conf(conf.body, 'base'))
                point[2] = 0
                point += 1e-2*np.array([0, 0, 1])
                if not (points and np.allclose(points[-1], point, atol=1e-3, rtol=0)):
                    points.append(point)
        points = get_target_path(self)
        return waypoints_from_path(points)

    def distance(self, distance_fn=get_distance):
        total = 0.
        for q1, q2 in zip(self.path, self.path[1:]):
            total += distance_fn(q1.values, q2.values)
        return total

    def iterate(self):
        for conf in self.path:
            yield conf

    def reverse(self):
        return Trajectory(reversed(self.path))

    def __repr__(self):
        d = 0
        if self.path:
            conf = self.path[0]
            d = 3 if isinstance(conf, Pose) else len(conf.joints)
        return 't({},{})'.format(d, len(self.path))

def create_trajectory(robot, joints, path):
    return Trajectory(Conf(robot, joints, q) for q in path)

##################################### Actions

class GripperCommand(Command):
    def __init__(self, robot, arm, position, teleport=False):
        self.robot = robot
        self.arm = arm
        self.position = position
        self.teleport = teleport

    def apply(self, state, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        start_conf = get_joint_positions(self.robot, joints)
        end_conf = [self.position] * len(joints)
        if self.teleport:
            path = [start_conf, end_conf]
        else:
            extend_fn = get_extend_fn(self.robot, joints)
            path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(self.robot, joints, positions)
            yield positions

    def control(self, **kwargs):
        joints = get_gripper_joints(self.robot, self.arm)
        positions = [self.position]*len(joints)
        control_mode = p.TORQUE_CONTROL
        for _ in joint_controller_hold(self.robot, joints, control_mode, positions):
            yield

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, self.position)

class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, HSR_TOOL_FRAMES.get(self.arm, self.arm))

    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)

    def apply(self, state, **kwargs):
        state.attachments[self.body] = create_attachment(self.robot, self.link, self.body)
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield

    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
        else:
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, HSR_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints]
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, HSR_TOOL_FRAMES.get(self.arm, self.arm))

    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        state.poses[self.body] = Pose(self.body, get_pose(self.body))
        del state.grasps[self.body]
        yield

    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)

    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

##################################### Additional Actions

class Clean(Command):
    def __init__(self, body):
        self.body = body

    def apply(self, state, **kwargs):
        state.cleaned.add(self.body)
        self.control()
        yield

    def control(self, **kwargs):
        p.addUserDebugText('Cleaned', textPosition=(0, 0, .25), textColorRGB=(0,0,1),
                           lifeTime=0, parentObjectUniqueId=self.body)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

class Cook(Command):
    def __init__(self, body):
        self.body = body

    def apply(self, state, **kwargs):
        state.cleaned.remove(self.body)
        state.cooked.add(self.body)
        self.control()
        yield

    def control(self, **kwargs):
        p.addUserDebugText('Cooked', textPosition=(0, 0, .5), textColorRGB=(1,0,0),
                           lifeTime=0, parentObjectUniqueId=self.body)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

##################################### Streams

def get_stable_gen(problem, collisions=True, **kwargs):
    # Sample place pose
    obstacles = problem.fixed if collisions else []
    extra = 'place'
    def gen(body, surface):
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces)
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface, extra)
            p.assign()
            # If the obstacles is not included in the surface and body, check pairwise collision between body.
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    return gen

def get_insert_gen(problem, collisions=True, **kwargs):
    # Sample insert pose
    obstacles = problem.fixed if collisions else []
    extra = 'insert'
    def gen(body, hole):
        if hole is None:
            holes = problem.holes
        else:
            holes = [hole]
        while True:
            hole = random.choice(holes)
            body_pose = sample_insertion(body, hole, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, hole, extra)
            p.assign()
            # If the obstacles is not included in the surface and body, check pairwise collision between body.
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, hole}):
                yield (p,)
    return gen

def get_grasp_gen(problem, collisions=True, randomize=False):
    # Sample pick pose
    for grasp_type in problem.grasp_types:
        if grasp_type not in GET_GRASPS:
            raise ValueError('Unexpected grasp type:', grasp_type)
    def fn(body):
        grasps = []
        arm = 'arm'
        if 'top' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE * get_unit_vector([1, 0, 0])
            grasps.extend(Grasp('top', body, g, multiply((approach_vector, unit_quat()), g), TOP_HOLDING_ARM)
                          for g in get_top_grasps(body, grasp_length=GRASP_LENGTH))
        if 'side' in problem.grasp_types:
            approach_vector = APPROACH_DISTANCE * get_unit_vector([0, -1, 0])
            grasps.extend(Grasp('side', body, g, multiply((approach_vector, unit_quat()), g), SIDE_HOLDING_ARM)
                          for g in get_side_grasps(body, grasp_length=GRASP_LENGTH))

        filtered_grasps = []
        for grasp in grasps:
            grasp_width = compute_grasp_width(problem.robot, arm, body, grasp.value) if collisions else 0.125 # TODO: modify
            if grasp_width is not None:
                grasp.grasp_width = grasp_width
                filtered_grasps.append(grasp)
        if randomize:
            random.shuffle(filtered_grasps)
        return [(g,) for g in filtered_grasps]
    return fn

def get_tool_from_root(robot, arm):
    root_link = link_from_name(robot, HSR_GRIPPER_ROOTS[arm])
    tool_link = link_from_name(robot, HSR_TOOL_FRAMES[arm])
    return get_relative_pose(robot, root_link, tool_link)

def iterate_approach_path(robot, arm, gripper, pose, grasp, body=None):
    tool_from_root = get_tool_from_root(robot, arm)
    grasp_pose = multiply(pose.value, invert(grasp.value))
    approach_pose = multiply(pose.value, invert(grasp.approach))
    for tool_pose in interpolate_poses(grasp_pose, approach_pose):
        set_pose(gripper, multiply(tool_pose, tool_from_root))
        if body is not None:
            set_pose(body, multiply(tool_pose, grasp.value))
        yield

def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, learned=False):
    # Sample move_base pose
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return

        gripper_pose = pose.value # multiply(pose.value, invert(grasp.value))
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')

        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            for base_conf in islice(base_generator, max_attempts):
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                yield (bq,)
                break
            else:
                yield None
    return gen_fn

def get_ik_fn(problem, custom_limits={}, collisions=True, teleport=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')

    saved_place_conf = {}
    saved_default_conf = {}
    def fn(arm, obj, pose, grasp, base_conf):
        # Obstacles
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}

        # HSR joints
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_base_joints(robot, arm)
        base_arm_joints = get_base_arm_joints(robot, arm)

        # Select grasp_type
        if 'shaft' in str(problem.body_names[obj]):
            grasp_type = 'side'
        elif 'gear' in str(problem.body_names[obj]):
            grasp_type = 'bottom'

        # Set planning parameters
        resolutions = 0.05 * np.ones(len(base_arm_joints))
        weights = [10, 10, 10, 50, 50, 50, 50, 50]

        if pose.extra == 'place':
            # Default confs
            default_arm_conf = get_carry_conf(arm, grasp_type)
            default_base_conf = get_joint_positions(robot, base_joints)
            default_base_arm_conf = [*default_base_conf, *default_arm_conf]

            pose.assign()
            base_conf.assign()

            attachment = grasp.get_attachment(problem.robot, arm)
            attachments = {attachment.child: attachment}

            # Get place pose and approach pose from deterministic_place
            place_pose, approach_pose = deterministic_place(obj, pose)

            # Set position to default configuration for grasp action
            set_joint_positions(robot, base_arm_joints, default_base_arm_conf)

            place_conf = hsr_inverse_kinematics(robot, arm, place_pose, custom_limits=custom_limits)
            if (place_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
                return None

            approach_conf = hsr_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits)
            if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                return None

            approach_conf = get_joint_positions(robot, base_arm_joints)

            # Plan joint motion for grasp
            place_path = plan_joint_motion(robot,
                                           base_arm_joints,
                                           place_conf,
                                           attachments=attachments.values(),
                                           obstacles=approach_obstacles,
                                           self_collisions=SELF_COLLISIONS,
                                           custom_limits=custom_limits,
                                           resolutions=resolutions/2.,
                                           weights=weights,
                                           restarts=2,
                                           iterations=25,
                                           smooth=25)
            if place_path is None:
                print('Place path failure')
                return None

            set_joint_positions(robot, base_arm_joints, default_base_arm_conf)
            saved_default_conf[str(obj)] = default_base_conf

            # Plan joint motion for approaching
            approach_path = plan_direct_joint_motion(robot,
                                                     base_arm_joints,
                                                     approach_conf,
                                                     attachments=attachments.values(),
                                                     obstacles=obstacles,
                                                     self_collisions=SELF_COLLISIONS,
                                                     custom_limits=custom_limits,
                                                     resolutions=resolutions)
            if approach_path is None:
                print('Approach path failure')
                return None

            # Save place conf
            set_joint_positions(robot, base_arm_joints, place_conf)
            saved_place_conf[str(obj)] = place_conf

            path1 = approach_path
            mt1 = create_trajectory(robot, base_arm_joints, path1)
            path2 = place_path
            mt2 = create_trajectory(robot, base_arm_joints, path2)
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt1, mt2])
            return (cmd,)

        elif pose.extra == 'insert':
            # Default confs
            default_arm_conf = arm_conf(arm, grasp.carry)
            default_base_conf = get_joint_positions(robot, base_joints)
            default_base_arm_conf = [*default_base_conf, *default_arm_conf]

            pose.assign()
            base_conf.assign()

            attachment = grasp.get_attachment(problem.robot, arm)
            attachments = {attachment.child: attachment}

            # Set position to default configuration for grasp action
            set_joint_positions(robot, base_arm_joints, default_base_arm_conf)

            # Get insert pose and depart pose from deterministic_insert
            insert_pose, depart_pose = deterministic_insert(obj, pose)
            depart_conf = hsr_inverse_kinematics(robot, arm, depart_pose, custom_limits=custom_limits)
            if (depart_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
                return None

            insert_conf = hsr_inverse_kinematics(robot, arm, insert_pose, custom_limits=custom_limits)
            if (insert_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
                return None

            # Set joints to place_conf
            place_conf = saved_place_conf[str(obj)]
            set_joint_positions(robot, base_arm_joints, place_conf)

            # Plan joint motion for insert action
            insert_path = plan_joint_motion(robot,
                                            base_arm_joints,
                                            insert_conf,
                                            attachments=attachments.values(),
                                            obstacles=approach_obstacles,
                                            self_collisions=SELF_COLLISIONS,
                                            custom_limits=custom_limits,
                                            resolutions=resolutions/2.,
                                            weights=weights,
                                            restarts=2,
                                            iterations=25,
                                            smooth=25)
            if insert_path is None:
                print('Insert path failure')
                return None

            # Set joints to insert_conf
            set_joint_positions(robot, base_arm_joints, insert_conf)

            # Plan joint motion for depart action
            depart_path = plan_direct_joint_motion(robot,
                                                   base_arm_joints,
                                                   depart_conf,
                                                   attachments=attachments.values(),
                                                   obstacles=obstacles,
                                                   self_collisions=SELF_COLLISIONS,
                                                   custom_limits=custom_limits,
                                                   resolutions=resolutions)
            if depart_path is None:
                print('Depart path failure')
                return None

            # Get end position to return
            return_conf = get_joint_positions(robot, base_arm_joints)

            # Set end positions to return
            default_arm_conf = arm_conf(arm, grasp.carry)
            default_base = saved_default_conf[str(obj)]
            default_base_arm_conf = [*default_base[:3], *default_arm_conf]
            set_joint_positions(robot, base_arm_joints, default_base_arm_conf)

            # Plan joint motion for return action
            return_path = plan_direct_joint_motion(robot,
                                                   base_arm_joints,
                                                   return_conf,
                                                   attachments=attachments.values(),
                                                   obstacles=obstacles,
                                                   self_collisions=SELF_COLLISIONS,
                                                   custom_limits=custom_limits,
                                                   resolutions=resolutions)
            if return_path is None:
                print('Return path failure')
                return None

            path1 = insert_path
            mt1 = create_trajectory(robot, base_arm_joints, path1)
            path2 = depart_path
            mt2 = create_trajectory(robot, base_arm_joints, path2)
            path3 = return_path
            mt3 = create_trajectory(robot, base_arm_joints, path3)
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt1, mt2, mt3])
            return (cmd,)

        else:
            # Default confs
            default_arm_conf = arm_conf(arm, grasp.carry)
            default_base_conf = get_joint_positions(robot, base_joints)
            default_base_arm_conf = [*default_base_conf, *default_arm_conf]

            pose.assign()
            base_conf.assign()

            attachment = grasp.get_attachment(problem.robot, arm)
            attachments = {attachment.child: attachment}

            # If pick action, return grasp pose (pose is obtained from PyBullet at initialization)
            pick_pose, approach_pose = deterministic_pick(obj, pose)

            # Set position to default configuration for grasp action
            set_joint_positions(robot, arm_joints, default_arm_conf)

            pick_conf = hsr_inverse_kinematics(robot, arm, pick_pose, custom_limits=custom_limits)
            if (pick_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):
                return None

            approach_conf = hsr_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits)
            if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                return None

            approach_conf = get_joint_positions(robot, base_arm_joints)

            # Plan joint motion for grasp
            grasp_path = plan_joint_motion(robot,
                                           base_arm_joints,
                                           pick_conf,
                                           attachments=attachments.values(),
                                           obstacles=approach_obstacles,
                                           self_collisions=SELF_COLLISIONS,
                                           custom_limits=custom_limits,
                                           resolutions=resolutions/2.,
                                           weights=weights,
                                           restarts=2,
                                           iterations=25,
                                           smooth=25)
            if grasp_path is None:
                print('Grasp path failure')
                return None

            set_joint_positions(robot, base_arm_joints, default_base_arm_conf)

            # Plan joint motion for approach
            approach_path = plan_direct_joint_motion(robot,
                                                     base_arm_joints,
                                                     approach_conf,
                                                     attachments=attachments.values(),
                                                     obstacles=obstacles,
                                                     self_collisions=SELF_COLLISIONS,
                                                     custom_limits=custom_limits,
                                                     resolutions=resolutions)
            if approach_path is None:
                print('Approach path failure')
                return None

            grasp_arm_conf = get_carry_conf(arm, grasp_type)
            grasp_base_conf = default_base_conf
            grasp_base_arm_conf = [*grasp_base_conf, *grasp_arm_conf]

            set_joint_positions(robot, base_arm_joints, approach_conf)

            # Plan joint motion for return
            return_path = plan_direct_joint_motion(robot,
                                                   base_arm_joints,
                                                   grasp_base_arm_conf,
                                                   attachments=attachments.values(),
                                                   obstacles=obstacles,
                                                   self_collisions=SELF_COLLISIONS,
                                                   custom_limits=custom_limits,
                                                   resolutions=resolutions)
            if return_path is None:
                print('Return path failure')
                return None

            path1 = approach_path
            mt1 = create_trajectory(robot, base_arm_joints, path1)
            path2 = grasp_path
            mt2 = create_trajectory(robot, base_arm_joints, path2)
            path3 = return_path
            mt3 = create_trajectory(robot, base_arm_joints, path3)
            cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt1, mt2, mt3])
            return (cmd,)
    return fn

def get_ik_ir_gen(problem, max_attempts=25, learned=False, teleport=False, **kwargs):
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=1, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
    return gen

def get_motion_gen(problem, custom_limits={}, collisions=True, teleport=False, motion_plan=False):
    robot = problem.robot
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    def fn(bq1, bq2, fluents=[]):
        saver.restore()
        bq1.assign()
        if teleport:
            path = [bq1, bq2]
        else:
            resolutions = 0.05 * np.ones(len(bq2.joints))
            raw_path = plan_joint_motion(robot,
                                         bq2.joints,
                                         bq2.values,
                                         attachments=[],
                                         obstacles=obstacles,
                                         custom_limits=custom_limits,
                                         self_collisions=SELF_COLLISIONS,
                                         resolutions=resolutions/2.,
                                         restarts=2,
                                         iterations=25,
                                         smooth=25)
            if raw_path is None:
                print('Failed motion plan!')
                return None
            path = [Conf(robot, bq2.joints, q) for q in raw_path]

        if motion_plan:
            goal_conf = base_values_from_pose(bq2.value)
            raw_path = plan_base_motion(robot, goal_conf, BASE_LIMITS, obstacles=obstacles)
            if raw_path is None:
                print('Failed motion plan!')
                return None

            path = [Pose(robot, pose_from_base_values(q, bq1.value)) for q in raw_path]
        bt = Trajectory(path)
        cmd = Commands(State(), savers=[BodySaver(robot)], commands=[bt])
        return (cmd,)
    return fn

def accelerate_gen_fn(gen_fn, max_attempts=1):
    def new_gen_fn(*inputs):
        generator = gen_fn(*inputs)
        while True:
            for i in range(max_attempts):
                try:
                    output = next(generator)
                except StopIteration:
                    return
                if output is not None:
                    print(gen_fn.__name__, i)
                    yield output
                    break
    return new_gen_fn

##################################################

def visualize_traj(path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    x_traj = path[0]
    y_traj = path[1]
    z_traj = path[2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x_traj, y_traj, z_traj, markersize=5, color='blue', marker='.')

    plt.show()

##################################################

def apply_commands_with_visualization(state, commands, time_step=None, pause=False, **kwargs):
    x_traj, y_traj, z_traj = [], [], []
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
            ee_pose = get_link_pose(2, link_from_name(2, HSR_TOOL_FRAMES['arm'])) # hsr is 2 (depends on problem)
            x_traj.append(ee_pose[0][0])
            y_traj.append(ee_pose[0][1])
            z_traj.append(ee_pose[0][2])
        if pause:
            wait_if_gui()

    path = [x_traj, y_traj, z_traj]
    visualize_traj(path)

def apply_commands_with_save(state, commands, time_step=None, pause=False, **kwargs):
    ee_traj = []
    joint_traj = []
    movable_joints = get_base_arm_joints(2, 'arm') # 8 joints
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
            ee_pose = get_link_pose(2, link_from_name(2, HSR_TOOL_FRAMES['arm']))
            ee_traj.append(ee_pose)

            joint_pos = get_joint_positions(2, movable_joints)
            joint_traj.append(joint_pos)

        if pause:
            wait_if_gui()

    np.save('simulation_ee_traj', ee_traj)
    np.save('simulation_joint_traj', joint_traj)

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    for i, command in enumerate(commands):
        print(i, command)
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

def apply_named_commands(state, commands, time_step=None, pause=False, **kwargs):
    for action_name, target_object_name, exp_command in commands:
        print(action_name, target_object_name, exp_command)
        for command in exp_command:
            for j, _ in enumerate(command.apply(state, **kwargs)):
                state.assign()
                if j == 0:
                    continue
                if time_step is None:
                    wait_for_duration(1e-2)
                    wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
                else:
                    wait_for_duration(time_step)
            if pause:
                wait_if_gui()

##################################### Debug

def control_commands(commands, **kwargs):
    wait_if_gui('Control?')
    disable_real_time()
    enable_gravity()
    for i, command in enumerate(commands):
        print(i, command)
        command.control(*kwargs)

##################################### Test

def get_target_point(conf):
    robot = conf.body
    link = link_from_name(robot, 'torso_lift_link')

    with BodySaver(conf.body):
        conf.assign()
        lower, upper = get_aabb(robot, link)
        center = np.average([lower, upper], axis=0)
        point = np.array(get_group_conf(conf.body, 'base'))
        point[2] = center[2]
        return point

def get_target_path(trajectory):
    return [get_target_point(conf) for conf in trajectory.path]

def get_cfree_pose_pose_test(collisions=True, **kwargs):
    def test(b1, p1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p1.assign()
        p2.assign()
        return not pairwise_collision(b1, b2, **kwargs)
    return test

def get_cfree_obj_approach_pose_test(collisions=True):
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        grasp_pose = multiply(p1.value, invert(g1.value))
        approach_pose = multiply(p1.value, invert(g1.approach), g1.value)
        for obj_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(b1, obj_pose)
            if pairwise_collision(b1, b2):
                return False
        return True
    return test

def get_cfree_approach_pose_test(problem, collisions=True):
    arm = 'arm'
    gripper = problem.get_gripper()
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        for _ in iterate_approach_path(problem.robot, arm, gripper, p1, g1, body=b1):
            if pairwise_collision(b1, b2) or pairwise_collision(gripper, b2):
                return False
        return True
    return test

def get_cfree_traj_pose_test(problem, collisions=True):
    def test(c, b2, p2):
        if not collisions:
            return True
        state = c.assign()
        if b2 in state.attachments:
            return True
        p2.assign()
        for _ in c.apply(state):
            state.assign()
            for b1 in state.attachments:
                if pairwise_collision(b1, b2):
                    return False
            if pairwise_collision(problem.robot, b2):
                return False
        return True
    return test

def get_cfree_traj_grasp_pose_test(problem, collisions=True):
    def test(c, a, b1, g, b2, p2):
        if not collisions or (b1 == b2):
            return True
        state = c.assign()
        if (b1 in state.attachments) or (b2 in state.attachments):
            return True
        p2.assign()
        grasp_attachment = g.get_attachment(problem.robot, a)
        for _ in c.apply(state):
            state.assign()
            grasp_attachment.assign()
            if pairwise_collision(b1, b2):
                return False
            if pairwise_collision(problem.robot, b2):
                return False
        return True
    return test

def get_supported(problem, collisions=True):
    def test(b, p1, r, p2):
        return is_placement(b, r)
    return test

def get_inserted(problem, collisions=True):
    def test(b, p1, r, p2):
        return is_inserted(b, r)
    return test

BASE_CONSTANT = 1
BASE_VELOCITY = 0.25

def distance_fn(q1, q2):
    distance = get_distance(q1.values[:2], q2.values[:2])
    return BASE_CONSTANT + distance / BASE_VELOCITY

def move_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY

##################################### Dataset

def calculate_delta_angular(q1, q2):
    # Transform
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)

    # Calculate difference
    diff = r2 * r1.inv()

    # Transform difference to euler 'xyz'
    delta_euler = diff.as_euler('xyz')

    return delta_euler.tolist()

def create_dataset(problem_name, robot, objects, object_names, state, commands, time_step=None, pause=False, **kwargs):
    movable_joints = get_base_arm_joints(robot, 'arm') # 8 joints
    gripper_joints = get_gripper_joints(robot, 'arm') # 2 joints
    whole_body_joints = movable_joints + gripper_joints # 10 joints
    full_joint_names = ['joint_x', 'joint_y', 'joint_rz', 'arm_lift_joint', 'torso_lift_joint',
                        'arm_flex_joint', 'head_pan_joint', 'arm_roll_joint', 'head_tilt_joint',
                        'wrist_flex_joint', 'wrist_roll_joint', 'hand_l_proximal_joint',
                        'hand_r_proximal_joint', 'hand_l_distal_joint', 'hand_r_distal_joint']
    full_joints = joints_from_names(robot, full_joint_names) # 15 joints

    # Up to what point should I record?
    skill_clip_index = {'move_base': 0, 'pick': 1, 'place': 1, 'insert': 2}

    # Prepare trajectory dict for all pose / positions
    traj = {'robot_pose': [], 'gripper_pose': [], 'ee_pose': [],
            'diff_robot_pose': [], 'diff_gripper_pose': [], 'diff_ee_pose': [],
            'object_pose': {object_names[obj]: list() for obj in objects}}

    # Prepare metadata for robot poses and object poses
    metadata = {'robot_init_pose': [],   # robot initial joint positions
                'robot_goal_pose': [],   # robot final joint positions
                'object_init_pose': {object_names[obj]: list() for obj in objects}, # object initial pose
                'object_goal_pose': [], # object goal pose
                'target_object_name': [], # pick target object's name
                'skill_name': []}

    goal_robot_poses = []
    goal_object_poses = []

    # Add metadata (init_data)
    metadata['robot_init_pose'] = get_joint_positions(robot, full_joints)
    for obj in objects:
        metadata['object_init_pose'][object_names[obj]] = get_pose(obj)

    for action_name, target_object_name, command in commands:
        # Get previous robot joint positions and end effector pose
        prev_robot_pose = get_joint_positions(robot, whole_body_joints)
        prev_gripper_pose = get_joint_positions(robot, gripper_joints)
        prev_ee_pose = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm']))

        command_num = 0

        # Execute commands
        for c in command:
            for j, _ in enumerate(c.apply(state, **kwargs)):
                state.assign()
                if j == 0:
                    continue
                if time_step is None:
                    wait_for_duration(1e-2)
                else:
                    wait_for_duration(time_step)

                # Get robot poses (10)
                robot_pose = get_joint_positions(robot, whole_body_joints)
                # Get gripper_joint poses (2)
                gripper_pose = get_joint_positions(robot, gripper_joints)
                # Get end effector poses (7)
                ee_pose = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm']))
                # Get diff robot poses (10)
                diff_robot_pose = tuple(curr_pose - prev_pose for curr_pose, prev_pose in zip(robot_pose, prev_robot_pose))
                # Get diff gripper_joint poses (2)
                diff_gripper_pose = tuple(curr_pose - prev_pose for curr_pose, prev_pose in zip(gripper_pose, prev_gripper_pose))
                # Get diff end_effector poses (6)
                diff_ee_pose = tuple([curr_pos - prev_pos for curr_pos, prev_pos in zip(ee_pose[0], prev_ee_pose[0])] + 
                                    calculate_delta_angular(ee_pose[1], prev_ee_pose[1]))

                # Absolute pose / positions
                traj['robot_pose'].append(robot_pose)
                traj['gripper_pose'].append(gripper_pose)
                traj['ee_pose'].append(ee_pose)
                # Relative pose / positions
                traj['diff_robot_pose'].append(diff_robot_pose)
                traj['diff_gripper_pose'].append(diff_gripper_pose)
                traj['diff_ee_pose'].append(diff_ee_pose)

                for obj in objects:
                    object_name = object_names[obj]
                    object_pose = get_pose(obj)
                    traj['object_pose'][object_name].append(object_pose)

                prev_robot_pose = robot_pose
                prev_gripper_pose = gripper_pose
                prev_ee_pose = ee_pose

                # Target object name
                metadata['target_object_name'].append(target_object_name)
                metadata['skill_name'].append(action_name)

            # Save each skill's target robot pose
            if skill_clip_index[action_name] == command_num:
                goal_robot_poses.append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))

            command_num += 1

        for obj in objects:
            if target_object_name == object_names[obj]:
                goal_object_poses.append(get_pose(obj))
        if action_name == 'move_base':
            goal_object_poses.append(get_pose(0))

    # Create dataset
    trajectory_path = os.path.join(os.environ['PYTHONPATH'].split(':')[1], 'experiments', 'env_3d',
                                    'dataset', problem_name, 'full', 'trajectory')
    num_traj_files = sum(os.path.isfile(os.path.join(trajectory_path, _name)) for _name in os.listdir(trajectory_path))

    metadata_path = os.path.join(os.environ['PYTHONPATH'].split(':')[1], 'experiments', 'env_3d',
                                    'dataset', problem_name, 'full', 'metadata')
    num_meta_files = sum(os.path.isfile(os.path.join(metadata_path, _name)) for _name in os.listdir(metadata_path))

    # File names
    traj_name = 'trajectory_' + str(num_traj_files) + '.json'
    metadata_name = 'metadata_' + str(num_meta_files) + '.json'

    # File paths
    traj_path = os.path.join(trajectory_path, traj_name)
    metadata_path = os.path.join(metadata_path, metadata_name)

    # Save trajectory / metadata
    with open(traj_path, 'w') as f:
        json.dump(traj, f)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def create_skill_dataset(problem_name, robot, objects, object_names, state, commands, time_step=None, pause=False, **kwargs):
    movable_joints = get_base_arm_joints(robot, 'arm') # 8 joints
    gripper_joints = get_gripper_joints(robot, 'arm') # 2 joints
    whole_body_joints = movable_joints + gripper_joints # 10 joints
    full_joint_names = ['joint_x', 'joint_y', 'joint_rz', 'arm_lift_joint', 'torso_lift_joint',
                        'arm_flex_joint', 'head_pan_joint', 'arm_roll_joint', 'head_tilt_joint',
                        'wrist_flex_joint', 'wrist_roll_joint', 'hand_l_proximal_joint',
                        'hand_r_proximal_joint', 'hand_l_distal_joint', 'hand_r_distal_joint']
    full_joints = joints_from_names(robot, full_joint_names) # 15 joints

    # Up to what point should I record?
    skill_record_index = {'move_base': [0, 0], 'pick': [0, 5], 'place': [0, 1], 'insert': [0, 2]}
    skill_clip_index = {'move_base': 0, 'pick': 1, 'place': 1, 'insert': 2}

    for action_name, target_object_name, command in commands:
        # Prepare trajectory dict for all pose / positions
        traj = {'robot_pose': [], 'gripper_pose': [], 'ee_pose': [],
                'diff_robot_pose': [], 'diff_gripper_pose': [], 'diff_ee_pose': [],
                'object_pose': {object_names[obj]: list() for obj in objects}}

        # Prepare metadata for robot poses and object poses
        metadata = {'robot_init_pose': [],   # robot initial joint positions
                    'robot_goal_pose': [],   # robot final joint positions
                    'robot_target_pose': [], # robot action target pose
                    'object_init_pose': {object_names[obj]: list() for obj in objects}, # object initial pose
                    'object_goal_pose': {object_names[obj]: list() for obj in objects}, # object goal pose
                    'target_object_name': [], # action target object's name
                    'fixed_object_name': []}  # object's name fixed to environment

        # Add metadata (init_data)
        metadata['robot_init_pose'] = get_joint_positions(robot, full_joints)
        for obj in objects:
            metadata['object_init_pose'][object_names[obj]] = get_pose(obj)
        metadata['target_object_name'] = target_object_name
        if target_object_name == 'gear1':
            metadata['fixed_object_name'] = ['shaft1', 'gearbox_base']
        elif target_object_name == 'gear2':
            metadata['fixed_object_name'] = ['shaft2', 'gearbox_base']
        else:
            metadata['fixed_object_name'] = ['gearbox_base']

        # Get previous robot joint positions and end effector pose
        prev_robot_pose = get_joint_positions(robot, whole_body_joints)
        prev_gripper_pose = get_joint_positions(robot, gripper_joints)
        prev_ee_pose = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm']))

        command_num = 0

        # Execute commands
        for c in command:
            for j, _ in enumerate(c.apply(state, **kwargs)):
                state.assign()
                if j == 0:
                    continue
                if time_step is None:
                    wait_for_duration(1e-2)
                else:
                    wait_for_duration(time_step)

                # Get robot poses (10)
                robot_pose = get_joint_positions(robot, whole_body_joints)
                # Get gripper_joint poses (2)
                gripper_pose = get_joint_positions(robot, gripper_joints)
                # Get end effector poses (7)
                ee_pose = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm']))
                # Get diff robot poses (10)
                diff_robot_pose = tuple(curr_pose - prev_pose for curr_pose, prev_pose in zip(robot_pose, prev_robot_pose))
                # Get diff gripper_joint poses (2)
                diff_gripper_pose = tuple(curr_pose - prev_pose for curr_pose, prev_pose in zip(gripper_pose, prev_gripper_pose))
                # Get diff end_effector poses (6)
                diff_ee_pose = tuple([curr_pos - prev_pos for curr_pos, prev_pos in zip(ee_pose[0], prev_ee_pose[0])] +
                                    calculate_delta_angular(ee_pose[1], prev_ee_pose[1]))

                if skill_record_index[action_name][0] <= command_num and \
                    skill_record_index[action_name][1] >= command_num:
                    # Absolute pose / positions
                    traj['robot_pose'].append(robot_pose)
                    traj['gripper_pose'].append(gripper_pose)
                    traj['ee_pose'].append(ee_pose)
                    # Relative pose / positions
                    traj['diff_robot_pose'].append(diff_robot_pose)
                    traj['diff_gripper_pose'].append(diff_gripper_pose)
                    traj['diff_ee_pose'].append(diff_ee_pose)

                    for obj in objects:
                        object_name = object_names[obj]
                        object_pose = get_pose(obj)
                        traj['object_pose'][object_name].append(object_pose)

                prev_robot_pose = robot_pose
                prev_gripper_pose = gripper_pose
                prev_ee_pose = ee_pose

            # Save each skill's target robot pose
            if skill_clip_index[action_name] == command_num:
                metadata['robot_target_pose'] = get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm']))

            command_num += 1

        if pause:
            wait_if_gui()

        # Add metadata (goal_data)
        metadata['robot_goal_pose'] = get_joint_positions(robot, full_joints)
        for obj in objects:
            metadata['object_goal_pose'][object_names[obj]] = get_pose(obj)

        dataset = [action_name, traj]

        # Create dataset
        trajectory_path = os.path.join(os.environ['PYTHONPATH'].split(':')[1], 'experiments', 'env_3d',
                                       'dataset', problem_name, action_name, 'trajectory')
        num_traj_files = sum(os.path.isfile(os.path.join(trajectory_path, _name)) for _name in os.listdir(trajectory_path))

        metadata_path = os.path.join(os.environ['PYTHONPATH'].split(':')[1], 'experiments', 'env_3d',
                                     'dataset', problem_name, action_name, 'metadata')
        num_meta_files = sum(os.path.isfile(os.path.join(metadata_path, _name)) for _name in os.listdir(metadata_path))

        # File names
        traj_name = 'trajectory_' + str(num_traj_files) + '.json'
        metadata_name = 'metadata_' + str(num_meta_files) + '.json'

        # File paths
        traj_path = os.path.join(trajectory_path, traj_name)
        metadata_path = os.path.join(metadata_path, metadata_name)

        # Save trajectory / metadata
        with open(traj_path, 'w') as f:
            json.dump(dataset, f)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

def replay_trajectory(robot, objects, object_names, state, commands, **kwargs):
    # Up to what point should I record?
    skill_clip_index = {'move_base': 0, 'pick': 0, 'place': 0, 'insert': 2}

    # Prepare trajectory dict for all pose / positions
    pick_metadata = {'init_robot_pose': [], 'goal_robot_pose': [], 'target_robot_pose': [],
                    'target_object_pose': []}
    place_metadata = {'init_robot_pose': [], 'goal_robot_pose': [], 'target_robot_pose': [],
                    'target_object_pose': []}
    insert_metadata = {'init_robot_pose': [], 'goal_robot_pose': [], 'target_robot_pose': [],
                    'target_object_pose': []}

    for action_name, target_object_name, command in commands:
        # Add metadata (init_data)
        if action_name == 'pick':
            pick_metadata['init_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
        elif action_name == 'place':
            place_metadata['init_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
        elif action_name == 'insert':
            insert_metadata['init_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))

        command_num = 0

        # Execute commands
        for c in command:
            for j, _ in enumerate(c.apply(state, **kwargs)):
                state.assign()
                wait_for_duration(1e-3)

            # Add metadata (target data)
            if skill_clip_index[action_name] == command_num:
                if action_name == 'pick':
                    pick_metadata['target_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
                elif action_name == 'place':
                    place_metadata['target_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
                elif action_name == 'insert':
                    insert_metadata['target_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))

            command_num += 1

        # Add metadata (goal_data)
        if action_name == 'pick':
            pick_metadata['goal_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
            for obj in objects:
                if object_names[obj] == target_object_name:
                    pick_metadata['target_object_pose'].append(get_pose(obj))
        elif action_name == 'place':
            place_metadata['goal_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
            for obj in objects:
                if object_names[obj] == target_object_name:
                    place_metadata['target_object_pose'].append(get_pose(obj))
        elif action_name == 'insert':
            insert_metadata['goal_robot_pose'].append(get_link_pose(robot, link_from_name(robot, HSR_TOOL_FRAMES['arm'])))
            for obj in objects:
                if object_names[obj] == target_object_name:
                    insert_metadata['target_object_pose'].append(get_pose(obj))

    return pick_metadata, place_metadata, insert_metadata

##################################### Custom

def deterministic_pick(pose, offset=[0.12, 0.14]):
    pick_pose = pose.value
    pick_pose = ((pick_pose[0][0]-offset[0], pick_pose[0][1], pick_pose[0][2]), pick_pose[1])
    approach_pose = ((pick_pose[0][0]-offset[1], pick_pose[0][1], pick_pose[0][2]), pick_pose[1])

    grasp_euler = np.array((np.pi/2, np.pi/2, 0.0))
    grasp_quat = R.from_euler('zyx', grasp_euler).as_quat()

    pick_pose = (pick_pose[0], grasp_quat)
    approach_pose = (approach_pose[0], grasp_quat)

    return pick_pose, approach_pose

def deterministic_place(pose, offset=[0.12, 0.0]):
    place_pose = pose.value
    place_pose = ((place_pose[0][0]-offset[0], place_pose[0][1], place_pose[0][2]), place_pose[1])
    approach_pose = ((place_pose[0][0]-offset[1], place_pose[0][1], place_pose[0][2]), place_pose[1])

    grasp_euler = np.array((np.pi/2, np.pi/2, 0.0))
    grasp_quat = R.from_euler('zyx', grasp_euler).as_quat()

    place_pose = (place_pose[0], grasp_quat)
    approach_pose = (approach_pose[0], grasp_quat)

    return place_pose, approach_pose

def deterministic_insert(pose, offset=[0.12, 0.14]):
    insert_pose = pose.value
    insert_pose = ((insert_pose[0][0]-offset[0], insert_pose[0][1], insert_pose[0][2]), insert_pose[1])
    depart_pose = ((insert_pose[0][0]-offset[1], insert_pose[0][1], insert_pose[0][2]), insert_pose[1])

    grasp_euler = np.array((np.pi/2, np.pi/2, 0.0))
    grasp_quat = R.from_euler('zyx', grasp_euler).as_quat()

    insert_pose = (insert_pose[0], grasp_quat)
    depart_pose = (depart_pose[0], grasp_quat)

    return insert_pose, depart_pose