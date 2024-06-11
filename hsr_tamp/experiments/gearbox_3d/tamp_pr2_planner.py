#!/usr/bin/env python

from __future__ import print_function

from utils.pybullet_tools.pr2_primitives import Pose, Conf, get_ik_ir_gen, get_motion_gen, \
    get_stable_gen, get_grasp_gen, Attach, Detach, Clean, Cook, control_commands, \
    get_gripper_joints, GripperCommand, apply_commands, State, get_cfree_approach_pose_test, \
    get_cfree_pose_pose_test, get_cfree_traj_pose_test, move_cost_fn
from utils.pybullet_tools.pr2_problems import cleaning_problem, cooking_problem, stacking_problem, holding_problem
from utils.pybullet_tools.pr2_utils import get_arm_joints, ARM_NAMES, get_group_joints, get_group_conf
from utils.pybullet_tools.utils import connect, get_pose, is_placement, point_from_pose, \
    disconnect, get_joint_positions, enable_gravity, save_state, restore_state, HideOutput, \
    get_distance, LockRenderer, get_min_limit, get_max_limit, has_gui, WorldSaver, wait_if_gui, add_line, SEPARATOR

from hsr_tamp.pddlstream.algorithms.meta import solve, create_parser
from hsr_tamp.pddlstream.language.generator import from_gen_fn, from_list_fn, from_fn, fn_from_constant, empty_gen, from_test
from hsr_tamp.pddlstream.language.constants import Equal, AND, print_solution, PDDLProblem
from hsr_tamp.pddlstream.language.function import FunctionInfo
from hsr_tamp.pddlstream.language.stream import StreamInfo, PartialInputs
from hsr_tamp.pddlstream.language.object import SharedOptValue
from hsr_tamp.pddlstream.language.external import defer_shared, never_defer
from hsr_tamp.pddlstream.utils import read, INF, get_file_path, find_unique, Profiler, str_from_object

from collections import namedtuple

BASE_CONSTANT = 1
BASE_VELOCITY = 0.5

#######################################################

def extract_point2d(v):
    if isinstance(v, Conf):
        return v.values[:2]
    if isinstance(v, Pose):
        return point_from_pose(v.value)[:2]
    if isinstance(v, SharedOptValue):
        if v.stream == 'sample-pose':
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

CustomValue = namedtuple('CustomValue', ['stream', 'values'])

def move_cost_fn(c):
    return 1

def opt_move_cost_fn(t):
    return 1

def opt_pose_fn(o, r):
    p = CustomValue('p-sp', (r,))
    return p,

def opt_ik_fn(a, o, p, g):
    q = CustomValue('q-ik', (p,))
    t = CustomValue('t-ik', tuple())
    return q, t

def opt_motion_fn(q1, q2):
    t = CustomValue('t-pbm', (q1, q2))
    return t,

#######################################################

class TAMPPlanner(object):

    def pddlstream_from_problem(self, problem, collisions=True, teleport=False):
        robot = problem.robot

        domain_pddl = read(get_file_path(__file__, 'task/cook/domain.pddl'))
        stream_pddl = read(get_file_path(__file__, 'task/cook/stream.pddl'))
        constant_map = {}

        initial_bq = Conf(robot, get_group_joints(robot, 'base'), get_group_conf(robot, 'base'))
        init = [
            ('CanMove',),
            ('BConf', initial_bq),
            ('AtBConf', initial_bq),
            Equal(('PickCost',), 1),
            Equal(('PlaceCost',), 1),
        ] + [('Sink', s) for s in problem.sinks] + \
            [('Stove', s) for s in problem.stoves] + \
            [('Connected', b, d) for b, d in problem.buttons] + \
            [('Button', b) for b, _ in problem.buttons]
        for arm in ARM_NAMES:
            joints = get_arm_joints(robot, arm)
            conf = Conf(robot, joints, get_joint_positions(robot, joints))
            init += [('Arm', arm), ('AConf', arm, conf), ('HandEmpty', arm), ('AtAConf', arm, conf)]
            if arm in problem.arms:
                init += [('Controllable', arm)]
        for body in problem.movable:
            pose = Pose(body, get_pose(body))
            init += [('Graspable', body), ('Pose', body, pose),
                    ('AtPose', body, pose)]
            for surface in problem.surfaces:
                init += [('Stackable', body, surface)]
                if is_placement(body, surface):
                    init += [('Supported', body, pose, surface)]

        goal = [AND]
        if problem.goal_conf is not None:
            goal_conf = Pose(robot, problem.goal_conf)
            init += [('BConf', goal_conf)]
            goal += [('AtBConf', goal_conf)]
        goal += [('Holding', a, b) for a, b in problem.goal_holding] + \
                        [('On', b, s) for b, s in problem.goal_on] + \
                        [('Cleaned', b)  for b in problem.goal_cleaned] + \
                        [('Cooked', b)  for b in problem.goal_cooked]

        stream_map = {
            'sample-pose': from_gen_fn(get_stable_gen(problem, collisions=collisions)),
            'sample-grasp': from_list_fn(get_grasp_gen(problem, collisions=False)),
            'inverse-kinematics': from_gen_fn(get_ik_ir_gen(problem, collisions=collisions, teleport=teleport)),
            'plan-base-motion': from_fn(get_motion_gen(problem, collisions=collisions, teleport=teleport)),
            'test-cfree-pose-pose': from_test(get_cfree_pose_pose_test(collisions=collisions)),
            'test-cfree-approach-pose': from_test(get_cfree_approach_pose_test(problem, collisions=collisions)),
            'test-cfree-traj-pose': from_test(get_cfree_traj_pose_test(problem.robot, collisions=collisions)),
            'MoveCost': move_cost_fn,
        }

        return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    def post_process(self, problem, plan, teleport=False):
        if plan is None:
            return None
        commands = []
        for i, (name, args) in enumerate(plan):
            if name == 'move_base':
                c = args[-1]
                new_commands = c.commands
            elif name == 'pick':
                a, b, p, g, _, c = args
                [t] = c.commands
                close_gripper = GripperCommand(problem.robot, a, g.grasp_width, teleport=teleport)
                attach = Attach(problem.robot, a, g, b)
                new_commands = [t, close_gripper, attach, t.reverse()]
            elif name == 'place':
                a, b, p, g, _, c = args
                [t] = c.commands
                gripper_joint = get_gripper_joints(problem.robot, a)[0]
                position = get_max_limit(problem.robot, gripper_joint)
                open_gripper = GripperCommand(problem.robot, a, position, teleport=teleport)
                detach = Detach(problem.robot, a, b)
                new_commands = [t, detach, open_gripper, t.reverse()]
            elif name == 'clean':
                body, sink = args
                new_commands = [Clean(body)]
            elif name == 'cook':
                body, stove = args
                new_commands = [Cook(body)]
            elif name == 'press_clean':
                body, sink, arm, button, bq, c = args
                [t] = c.commands
                new_commands = [t, Clean(body), t.reverse()]
            elif name == 'press_cook':
                body, sink, arm, button, bq, c = args
                [t] = c.commands
                new_commands = [t, Cook(body), t.reverse()]
            else:
                raise ValueError(name)
            print(i, name, args, new_commands)
            commands += new_commands
        return commands

    def plan(self, partial=False, defer=False, verbose=True):
        parser = create_parser()
        parser.add_argument('-cfree', action='store_true', help='Disables collisions during planning')
        parser.add_argument('-enable', action='store_true', help='Enables rendering during planning')
        parser.add_argument('-teleport', action='store_true', help='Teleports between configurations')
        parser.add_argument('-simulate', action='store_true', help='Simulates the system')
        args = parser.parse_args()
        print('Arguments:', args)

        connect(use_gui=True)
        problem_fn = stacking_problem
        with HideOutput():
            problem = problem_fn()
        saver = WorldSaver()

        pddlstream_problem = self.pddlstream_from_problem(problem, collisions=not args.cfree, teleport=args.teleport)

        stream_info = {
            'MoveCost': FunctionInfo(opt_move_cost_fn),
        }
        stream_info.update({
            'sample-pose': StreamInfo(opt_gen_fn=PartialInputs('?r')),
            'inverse-kinematics': StreamInfo(opt_gen_fn=PartialInputs('?p')),
            'plan-base-motion': StreamInfo(opt_gen_fn=PartialInputs('?q1 ?q2'), defer_fn=defer_shared if defer else never_defer),
        } if partial else {
            'sample-pose': StreamInfo(opt_gen_fn=from_fn(opt_pose_fn)),
            'inverse-kinematics': StreamInfo(opt_gen_fn=from_fn(opt_ik_fn)),
            'plan-base-motion': StreamInfo(opt_gen_fn=from_fn(opt_motion_fn)),
        })
        _, _, _, stream_map, init, goal = pddlstream_problem
        print('Init:', init)
        print('Goal:', goal)
        print('Streams:', str_from_object(set(stream_map)))
        print(SEPARATOR)

        with Profiler():
            with LockRenderer(lock=not args.enable):
                solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=args.unit,
                                stream_info=stream_info, success_cost=INF, verbose=True, debug=False)
                saver.restore()

        print_solution(solution)
        plan, cost, evaluations = solution
        if (plan is None) or not has_gui():
            disconnect()
            return

        print(SEPARATOR)
        with LockRenderer(lock=not args.enable):
            commands = self.post_process(problem, plan)
            problem.remove_gripper()
            saver.restore()

        saver.restore()
        wait_if_gui('Execute?')
        if args.simulate:
            control_commands(commands)
        else:
            apply_commands(State(), commands, time_step=0.01)
        wait_if_gui('Finish?')
        disconnect()

if __name__ == '__main__':
    tamp_planner = TAMPPlanner()
    tamp_planner.plan()