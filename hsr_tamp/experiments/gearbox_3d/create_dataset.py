#!/usr/bin/env python
import threading
import numpy as np
from collections import namedtuple

from utils.pybullet_tools.hsrb_problems import PROBLEMS
from utils.pybullet_tools.hsrb_utils import get_arm_joints, get_gripper_joints, get_group_joints, get_group_conf
from utils.pybullet_tools.utils import (
    # Simulation utility
    connect, disconnect, save_state, has_gui, restore_state, wait_if_gui, \
    # Getter
    get_pose, get_distance, get_joint_positions, get_max_limit, \
    is_placement, point_from_pose, \
    LockRenderer, WorldSaver, SEPARATOR)
from utils.pybullet_tools.hsrb_primitives import (
    # Geometry
    Pose, Conf, State, Cook, Clean, \
    # Command
    GripperCommand, Attach, Detach, \
    # Utility
    apply_commands, apply_named_commands, control_commands, \
    create_dataset, create_skill_dataset, \
    # Getter
    get_ik_ir_gen, get_motion_gen, get_stable_gen, get_grasp_gen, get_insert_gen, \
    # Tester
    get_cfree_approach_pose_test, get_cfree_pose_pose_test, get_cfree_traj_pose_test, \
    get_supported, get_inserted, \
    # Cost function
    move_cost_fn)

from hsr_tamp.pddlstream.algorithms.meta import solve, create_parser
from hsr_tamp.pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, fn_from_constant, empty_gen, from_test
from hsr_tamp.pddlstream.language.constants import print_solution, Equal, AND, PDDLProblem
from hsr_tamp.pddlstream.language.external import defer_shared, never_defer
from hsr_tamp.pddlstream.language.function import FunctionInfo
from hsr_tamp.pddlstream.language.stream import StreamInfo, PartialInputs
from hsr_tamp.pddlstream.language.object import SharedOptValue
from hsr_tamp.pddlstream.utils import find_unique, get_file_path, read, str_from_object, Profiler, INF


BASE_CONSTANT = 1
BASE_VELOCITY = 0.5

#######################################################

def extract_point2d(v):
    if isinstance(v, Conf):
        return v.values[:2]
    if isinstance(v, Pose):
        return point_from_pose(v.value)[:2]
    if isinstance(v, SharedOptValue):
        if v.stream == 'sample-place':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'sample-insert':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'inverse-kinematics':
            p, = v.values
            return extract_point2d(p)
    if isinstance(v, CustomValue):
        if v.stream == 'p-sp':
            r, = v.values
            return point_from_pose(get_pose(r))[:2]
        if v.stream == 'q-ik':
            p, = v.values
            return extract_point2d(p)
    raise ValueError(v.stream)

#######################################################

def select_plan(plan, movable):
    import itertools

    predefined_order = list(itertools.permutations(movable))
    filtered_permutations = [p for p in predefined_order if p.index(6) < p.index(3)]
    filtered_permutations = [p for p in filtered_permutations if p.index(7) < p.index(4)]
    filtered_permutations = [p for p in filtered_permutations if p.index(5) < p.index(4)]

    object_order = []
    for i, (name, args) in enumerate(plan):
        if name == 'insert':
            _, b1, b2, _, _, _, _, _, _ = args
            object_order.append(b1)
        else:
            pass

    object_order = tuple(object_order)

    executable = True
    if object_order in filtered_permutations:
        executable = True
    else:
        executable = False

    return executable

#######################################################

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def move_cost_fn(c):
    return 1

def opt_move_cost_fn(t):
    return 1

def opt_place_fn(o, r):
    p2 = CustomValue('p-sp', (r,))
    return p2,

def opt_insert_fn(o, r):
    p2 = CustomValue('p-si', (r,))
    return p2,

def opt_ik_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

class TAMPPlanner(object):

    def pddlstream_from_problem(self, problem, collisions=True, teleport=False, resolution=0.05):
        robot = problem.robot

        domain_pddl = read(get_file_path(__file__, 'task/assemble/domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'task/assemble/stream.pddl'))
        constant_map = {}

        initial_bq = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
        init = [
            ('CanMove',),
            ('BConf', initial_bq),
            ('AtBConf', initial_bq),
            Equal(('PickCost',), 1),
            Equal(('PlaceCost',), 1),
            Equal(('InsertCost',), 1),
        ] + [('Sink', s) for s in problem.sinks] + \
            [('Stove', s) for s in problem.stoves] + \
            [('Connected', b, d) for b, d in problem.buttons] + \
            [('Button', b) for b, _ in problem.buttons]

        joints = get_arm_joints(robot, 'arm')
        conf = Conf(robot, joints, get_joint_positions(robot, joints))
        init += [('Arm', 'arm'), ('AConf', 'arm', conf), ('HandEmpty', 'arm'), ('AtAConf', 'arm', conf)]
        init += [('Controllable', 'arm')]

        for body in problem.movable:
            pose = Pose(body, get_pose(body))
            init += [('Graspable', body), ('Pose', body, pose),
                    ('AtPose', body, pose)]

        goal = [AND]
        if problem.goal_conf is not None:
            goal_conf = Pose(robot, problem.goal_conf)
            init += [('BConf', goal_conf)]
            goal += [('AtBConf', goal_conf)]

        for body in problem.surfaces:
            pose = Pose(body, get_pose(body))
            init += [('RegionPose', body, pose)]

        for body in problem.holes:
            pose = Pose(body, get_pose(body))
            init += [('HolePose', body, pose)]

        init += [('Inserted', b1) for b1 in problem.holes]
        init += [('Placeable', b1, b2) for b1, b2 in problem.init_placeable]
        init += [('Insertable', b1, b2) for b1, b2 in problem.init_insertable]
        goal += [('Holding', a, b) for a, b in problem.goal_holding] + \
                [('On', a, b) for a, b in problem.goal_on] + \
                [('InHole', a, b) for a, b in problem.goal_inserted] + \
                [('Cleaned', b)  for b in problem.goal_cleaned] + \
                [('Cooked', b)  for b in problem.goal_cooked]

        stream_map = {
            'sample-place': from_gen_fn(get_stable_gen(problem, collisions=collisions)),
            'sample-insert': from_gen_fn(get_insert_gen(problem, collisions=collisions)),
            'sample-grasp': from_list_fn(get_grasp_gen(problem, collisions=False)),
            'plan-base-motion': from_fn(get_motion_gen(problem, collisions=True, teleport=teleport, resolution=resolution)),
            'inverse-kinematics': from_gen_fn(get_ik_ir_gen(problem, collisions=collisions, teleport=teleport, resolution=resolution)),
            'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
            'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
            'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem, collisions=collisions)),
            'test-supported': from_test(get_supported(problem, collisions=collisions)),
            'test-inserted': from_test(get_inserted(problem, collisions=collisions)),
            'MoveCost': move_cost_fn,
        }

        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def initialize(self, parser):
        parser.add_argument('-e', '--enable', action='store_true', help='Enables rendering during planning')
        parser.add_argument('-d', '--deterministic', action='store_true', help='Uses a deterministic sampler')
        parser.add_argument('-t', '--max_time', default=30, type=int, help='The max time')
        parser.add_argument('-c', '--cfree', default=True, type=bool, help="select collision activate")
        parser.add_argument('-n', '--number', default=4, type=int, help='The number of blocks')
        parser.add_argument('-p', '--problem', default='gearbox_problem', help='The name of the problem to solve')
        parser.add_argument('-v', '--visualize', action='store_true', help='Visualizes graphs')
        parser.add_argument("--gpu_id", help="select using gpu id", type=str, default="-1")
        parser.add_argument("--save", help="select save models", type=bool, default=True)
        parser.add_argument("--debug", help="save visualization", type=bool, default=False)
        parser.add_argument("--resolution", help="select resolution", type=float, default=0.005)
        parser.add_argument("--skill", action='store_true', help="select skill dataset")
        parser.add_argument("--teleport", action='store_true', help='Teleports between configurations')
        parser.add_argument("--simulate", action='store_true', help='Simulates the system')
        args = parser.parse_args()
        print('Arguments:', args)

        connect(use_gui=True, shadows=False)

        np.set_printoptions(precision=2)
        if args.deterministic:
            self.set_deterministic()

        problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
        if args.problem not in problem_from_name:
            raise ValueError(args.problem)
        print('Problem:', args.problem)
        problem_fn = problem_from_name[args.problem]
        tamp_problem = problem_fn()

        return tamp_problem, args
    
    def post_process(self, problem, plan, teleport=False):
        if plan is None:
            return None
        commands = []
        for i, (name, args) in enumerate(plan):
            if name == 'move_base':
                q1, q2, c = args
                target_object_name = None
                new_commands = c.commands
            elif name == 'pick':
                a, b, p, g, _, c = args
                [traj_approach, traj_pick, traj_return] = c.commands
                close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
                attach = Attach(problem.robot, a, g, b)
                target_object_name = problem.body_names[b]
                new_commands = [traj_approach, traj_pick, close_gripper, attach, traj_pick.reverse(), traj_return]
            elif name == 'place':
                a, b1, b2, p, g, _, c = args
                [traj_approach, traj_place] = c.commands
                gripper_joint = get_gripper_joints(problem.robot, a)[0]
                position = get_max_limit(problem.robot, gripper_joint)
                target_object_name = problem.body_names[b1]
                new_commands = [traj_approach, traj_place]
            elif name == 'insert':
                a, b1, b2, p1, p2, g, _, _, c = args
                [traj_insert, traj_depart, traj_return] = c.commands
                gripper_joint = get_gripper_joints(problem.robot, a)[0]
                position = get_max_limit(problem.robot, gripper_joint)
                open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
                detach = Detach(problem.robot, a, b1)
                target_object_name = problem.body_names[b1]
                new_commands = [traj_insert, detach, open_gripper, traj_depart, traj_return.reverse()]
            elif name == 'clean':
                body, sink = args
                new_commands = [Clean(body)]
            elif name == 'cook':
                body, stove = args
                new_commands = [Cook(body)]
            else:
                raise ValueError(name)
            print(i, name, args, new_commands)
            commands += ((name, target_object_name, new_commands),)
        return commands

    def plan(self):
        parser = create_parser()
        parser.add_argument('-g', '--gurobi', action='store_true', help='Uses gurobi')
        parser.add_argument('-o', '--optimal', action='store_true', help='Runs in an anytime mode')
        parser.add_argument('-s', '--skeleton', action='store_true', help='Enforces skeleton plan constraints')
        tamp_problem, args = self.initialize(parser)

        saver = WorldSaver()

        pddlstream_problem = self.pddlstream_from_problem(tamp_problem, collisions=not args.cfree,
                                                          teleport=args.teleport, resolution=args.resolution)

        stream_info = {
            'MoveCost': FunctionInfo(opt_move_cost_fn),
            'sample-place': StreamInfo(opt_gen_fn=from_fn(opt_place_fn)),
            'sample-insert': StreamInfo(opt_gen_fn=from_fn(opt_insert_fn)),
            'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn)),
            'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
        }

        _, _, _, stream_map, init, goal = pddlstream_problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))
        print(SEPARATOR)

        with Profiler():
            with LockRenderer(lock=not args.enable):
                solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
                                stream_info=stream_info, success_cost=INF, max_time=60,
                                max_iterations=4, max_complexity=4, verbose=True, debug=False)
                saver.restore()

        print_solution(solution)
        plan, cost, evaluations = solution

        print('#############################')
        print('plan: ', plan)
        print('#############################')

        if (plan is None):
            disconnect()
            return

        with LockRenderer(lock=not args.enable):
            commands = self.post_process(tamp_problem, plan)
            tamp_problem.remove_gripper()
            saver.restore()

        if args.problem == 'gearbox_problem':
            if not select_plan(plan, tamp_problem.movable):
                print("Execute Only")
                return
            else:
                print("Create Dataset!")
                if args.skill:
                    create_skill_dataset(args.problem, tamp_problem.robot, tamp_problem.bodies, tamp_problem.body_names, State(), commands, time_step=0.01)
                else:
                    create_dataset(args.problem, tamp_problem.robot, tamp_problem.bodies, tamp_problem.body_names, State(), commands, time_step=0.01)
        else:
            print("Create Dataset!")
            if args.skill:
                create_skill_dataset(args.problem, tamp_problem.robot, tamp_problem.bodies, tamp_problem.body_names, State(), commands, time_step=0.01)
            else:
                create_dataset(args.problem, tamp_problem.robot, tamp_problem.bodies, tamp_problem.body_names, State(), commands, time_step=0.01)

        disconnect()
        return

def main_loop():
    tamp_planner = TAMPPlanner()
    tamp_planner.plan()


if __name__ == '__main__':
    main_loop()

    # Threadding
    # thread1 = threading.Thread(target=main_loop)
    # thread2 = threading.Thread(target=main_loop)
    # thread3 = threading.Thread(target=main_loop)
    # thread4 = threading.Thread(target=main_loop)

    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()

    # thread1.join()
    # thread2.join()
    # thread3.join()
    # thread4.join()