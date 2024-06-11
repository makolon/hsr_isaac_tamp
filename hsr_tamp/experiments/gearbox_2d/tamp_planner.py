#!/usr/bin/env python3
import os
import time
import random
import numpy as np
from itertools import product

from utils.optimizer.optimizer import cfree_motion_fn, get_optimize_fn
from utils.primitives import (
    # generator
    get_pose_gen, get_stack_pose_gen, get_block_pose_gen, \
    get_center_pose_gen, get_stack_center_pose_gen, get_block_center_pose_gen, \
    # test function
    collision_test, stack_test, get_region_test, get_block_region_test, \
    # utility
    distance_fn, inverse_kin_fn, duration_fn, stack_inverse_kin_fn, \
    plan_motion, draw_state, get_random_seed, update_state, \
    # constant
    SUCTION_HEIGHT, MOVE_COST, GRASP, ENVIRONMENT_NAMES, PROBLEMS)

from hsr_tamp.pddlstream.algorithms.meta import solve, create_parser
from hsr_tamp.pddlstream.algorithms.downward import get_cost_scale
from hsr_tamp.pddlstream.algorithms.constraints import PlanConstraints, WILD
from hsr_tamp.pddlstream.algorithms.visualization import VISUALIZATIONS_DIR
from hsr_tamp.pddlstream.language.external import defer_shared, get_defer_all_unbound, get_defer_any_unbound
from hsr_tamp.pddlstream.language.constants import And, Equal, PDDLProblem, TOTAL_COST, print_solution, Or, Output
from hsr_tamp.pddlstream.language.function import FunctionInfo
from hsr_tamp.pddlstream.language.generator import from_gen_fn, from_list_fn, from_test, from_fn
from hsr_tamp.pddlstream.language.stream import StreamInfo
from hsr_tamp.pddlstream.language.temporal import get_end, compute_duration, retime_plan
from hsr_tamp.pddlstream.utils import ensure_dir, safe_rm_dir, user_input, read, INF, get_file_path, str_from_object, \
    sorted_str_from_list, implies, inclusive_range, Profiler


##################################################

TIGHT_SKELETON = [
    ('move', ['r0', '?q0', WILD, '?q1']),
    ('pick', ['r0', 'B', '?p0', '?g0', '?q1']),
    ('move', ['r0', '?q1', WILD, '?q2']),
    ('place', ['r0', 'B', '?p1', '?g0', '?q2']),

    ('move', ['r0', '?q2', WILD, '?q3']),
    ('pick', ['r0', 'A', '?p2', '?g1', '?q3']),
    ('move', ['r0', '?q3', WILD, '?q4']),
    ('place', ['r0', 'A', '?p3', '?g1', '?q4']),
    ('move', ['r0', '?q4', WILD, '?q5']),
]

MUTEXES = [
    #[('kin', '?b1', '?p1', '?q'), ('kin', '?b2', '?p2', '?q')],
]


class TAMPPlanner(object):

    def create_problem(self, tamp_problem, hand_empty=False, manipulate_cost=1.):
        initial = tamp_problem.initial
        assert(not initial.holding)

        init = [
            Equal(('Cost',), manipulate_cost),
            Equal((TOTAL_COST,), 0)] + [('Placeable', b, r) for b in initial.block_poses.keys()
                for r in tamp_problem.regions if (r in ENVIRONMENT_NAMES)]

        goal_literals = []

        ### Block specification
        for b, p in initial.block_poses.items():
            init += [
                ('Block', b),
                ('Pose', b, p),
                ('AtPose', b, p),
            ]

        for b, r in tamp_problem.goal_in.items():
            if isinstance(r, np.ndarray):
                init += [('Pose', b, r)]
                goal_literals += [('AtPose', b, r)]
            else:
                blocks = [b] if isinstance(b, str) else b
                regions = [r] if isinstance(r, str) else r
                conditions = []
                for body, region in product(blocks, regions):
                    init += [('Region', region), ('Placeable', body, region)]
                    conditions += [('In', body, region)]
                goal_literals.append(Or(*conditions))

        for on_list in tamp_problem.goal_on:
            init += [('Placeable', on_list[1], on_list[0])]
            goal_literals += [('On', on_list[1], on_list[0])]

        ### Robot specification
        for r, q in initial.robot_confs.items():
            init += [
                ('Robot', r),
                ('CanMove', r),
                ('Conf', q),
                ('AtConf', r, q),
                ('HandEmpty', r),
            ]
            if hand_empty:
                goal_literals += [('HandEmpty', r)]
            if tamp_problem.goal_conf is not None:
                goal_literals += [('AtConf', r, q)]

        goal = And(*goal_literals)

        return init, goal

    def pddlstream_from_tamp(self, tamp_problem, use_stream=True, use_optimizer=False, collisions=True, center=False):

        domain_pddl = read(get_file_path(__file__, 'task/stack/domain.pddl'))
        external_paths = []
        if use_stream:
            external_paths.append(get_file_path(__file__, 'task/stack/stream.pddl'))
        if use_optimizer:
            external_paths.append(get_file_path(__file__, 'optimizer/optimizer.pddl'))
        external_pddl = [read(path) for path in external_paths]

        constant_map = {}
        if center:
            stream_map = {
                's-grasp': from_fn(lambda b: (GRASP,)),
                's-motion': from_fn(plan_motion),
                's-ik': from_fn(inverse_kin_fn),
                's-stackik': from_fn(stack_inverse_kin_fn),
                's-region': from_gen_fn(get_center_pose_gen(tamp_problem.regions)),
                's-blockregion': from_gen_fn(get_block_center_pose_gen(tamp_problem.regions)),
                't-region': from_test(get_region_test(tamp_problem.regions)),
                't-blockregion': from_test(get_block_region_test(tamp_problem.regions)),
                't-cfree': from_test(lambda *args: implies(collisions, not collision_test(*args))),
                't-cstack': from_test(lambda *args: implies(collisions, not stack_test(*args))),
                'dist': distance_fn,
                'duration': duration_fn,
            }
        else:
            stream_map = {
                's-grasp': from_fn(lambda b: (GRASP,)),
                's-motion': from_fn(plan_motion),
                's-ik': from_fn(inverse_kin_fn),
                's-stackik': from_fn(stack_inverse_kin_fn),
                's-region': from_gen_fn(get_pose_gen(tamp_problem.regions)),
                's-blockregion': from_gen_fn(get_block_pose_gen(tamp_problem.regions)),
                't-region': from_test(get_region_test(tamp_problem.regions)),
                't-blockregion': from_test(get_block_region_test(tamp_problem.regions)),
                't-cfree': from_test(lambda *args: implies(collisions, not collision_test(*args))),
                't-cstack': from_test(lambda *args: implies(collisions, not stack_test(*args))),
                'dist': distance_fn,
                'duration': duration_fn,
            }

        if use_optimizer:
            stream_map.update({
                'gurobi': from_list_fn(get_optimize_fn(tamp_problem.regions, collisions=collisions)),
                'rrt': from_fn(cfree_motion_fn),
            })

        init, goal = self.create_problem(tamp_problem)

        return PDDLProblem(domain_pddl, constant_map, external_pddl, stream_map, init, goal)

    def display_plan(self, tamp_problem, plan, display=True, save=False, time_step=0.025, sec_per_step=1e-3):
        from utils.viewer import ContinuousTMPViewer, COLORS

        if save:
            example_name = 'continuous_tamp'
            directory = os.path.join(VISUALIZATIONS_DIR, '{}/'.format(example_name))
            safe_rm_dir(directory)
            ensure_dir(directory)

        colors = dict(zip(sorted(tamp_problem.initial.block_poses.keys()), COLORS))
        viewer = ContinuousTMPViewer(SUCTION_HEIGHT, tamp_problem.regions, title='Continuous TAMP')
        state = tamp_problem.initial
        print()
        print(state)
        duration = compute_duration(plan)
        real_time = None if sec_per_step is None else (duration * sec_per_step / time_step)
        print('Duration: {} | Step size: {} | Real time: {}'.format(duration, time_step, real_time))

        draw_state(viewer, state, colors)
        if display:
            user_input('Start?')
        if plan is not None:
            for t in inclusive_range(0, duration, time_step):
                for action in plan:
                    if action.start <= t <= get_end(action):
                        update_state(state, action, t - action.start)
                print('t={} | {}'.format(t, state))
                draw_state(viewer, state, colors)
                if save:
                    viewer.save(os.path.join(directory, 't={}'.format(t)))
                if display:
                    if sec_per_step is None:
                        user_input('Continue?')
                    else:
                        time.sleep(sec_per_step)
        if display:
            user_input('Finish?')
        return state

    def set_deterministic(self, seed=0):
        random.seed(seed=seed)
        np.random.seed(seed=seed)

    def initialize(self, parser):
        parser.add_argument('-c', '--cfree', action='store_true', help='Disables collisions')
        parser.add_argument('-d', '--deterministic', action='store_true', help='Uses a deterministic sampler')
        parser.add_argument('-t', '--max_time', default=30, type=int, help='The max time')
        parser.add_argument('-n', '--number', default=4, type=int, help='The number of blocks')
        parser.add_argument('-p', '--problem', default='gearbox', help='The name of the problem to solve')
        parser.add_argument('-v', '--visualize', action='store_true', help='Visualizes graphs')

        args = parser.parse_args()
        print('Arguments:', args)
        np.set_printoptions(precision=2)
        if args.deterministic:
            self.set_deterministic()
        print('Random seed:', get_random_seed())

        problem_from_name = {fn.__name__: fn for fn in PROBLEMS}
        if args.problem not in problem_from_name:
            raise ValueError(args.problem)
        print('Problem:', args.problem)
        problem_fn = problem_from_name[args.problem]
        tamp_problem = problem_fn(args.number)
        print(tamp_problem)
        return tamp_problem, args

    def dump_pddlstream(self, pddlstream_problem):
        print('Initial:', sorted_str_from_list(pddlstream_problem.init))
        print('Goal:', str_from_object(pddlstream_problem.goal))

    def plan(self):
        parser = create_parser()
        parser.add_argument('-g', '--gurobi', action='store_true', help='Uses gurobi')
        parser.add_argument('-o', '--optimal', action='store_true', help='Runs in an anytime mode')
        parser.add_argument('-s', '--skeleton', action='store_true', help='Enforces skeleton plan constraints')
        tamp_problem, args = self.initialize(parser)

        defer_fn = defer_shared # always True
        stream_info = {
            's-grasp': StreamInfo(defer_fn=defer_fn),
            's-region': StreamInfo(defer_fn=defer_fn),
            's-ik': StreamInfo(defer_fn=get_defer_all_unbound(inputs='?g')),
            's-motion': StreamInfo(defer_fn=get_defer_any_unbound()),
            't-region': StreamInfo(eager=False, p_success=0),
            't-blockregion': StreamInfo(eager=False, p_success=0),
            't-cfree': StreamInfo(defer_fn=get_defer_any_unbound(), eager=False),
            't-cstack': StreamInfo(eager=False),
            'gurobi-cfree': StreamInfo(eager=False, negate=True),
            'dist': FunctionInfo(eager=False, defer_fn=get_defer_any_unbound(), opt_fn=lambda q1, q2: MOVE_COST),
        }

        hierarchy = [
            # ABSTRIPSLayer(pos_pre=['atconf']), #, horizon=1),
        ]

        skeletons = [TIGHT_SKELETON] if args.skeleton else None
        assert implies(args.skeleton, args.problem == 'tight')
        max_cost = INF # 8*MOVE_COST
        constraints = PlanConstraints(skeletons=skeletons,
                                    exact=True,
                                    max_cost=max_cost)
        replan_actions = set()

        pddlstream_problem = self.pddlstream_from_tamp(tamp_problem, collisions=not args.cfree,
                                center=True, use_stream=not args.gurobi, use_optimizer=args.gurobi)
        self.dump_pddlstream(pddlstream_problem)

        success_cost = 0 if args.optimal else INF
        planner = 'max-astar'
        effort_weight = 1. / get_cost_scale()

        with Profiler(field='cumtime', num=20):
            solution = solve(pddlstream_problem, algorithm=args.algorithm, constraints=constraints, stream_info=stream_info,
                            replan_actions=replan_actions, planner=planner, max_planner_time=10, hierarchy=hierarchy,
                            max_time=args.max_time, max_iterations=INF, debug=False, verbose=True,
                            unit_costs=args.unit, success_cost=success_cost,
                            unit_efforts=True, effort_weight=effort_weight,
                            search_sample_ratio=1, visualize=args.visualize)

        print_solution(solution)
        plan, cost, evaluations = solution

        return plan, cost, evaluations

    def execute(self):
        parser = create_parser()
        parser.add_argument('-g', '--gurobi', action='store_true', help='Uses gurobi')
        parser.add_argument('-o', '--optimal', action='store_true', help='Runs in an anytime mode')
        parser.add_argument('-s', '--skeleton', action='store_true', help='Enforces skeleton plan constraints')
        tamp_problem, _ = self.initialize(parser)
        plan, _, _ = self.plan()
        print('plan: ', plan)
        if plan is not None:
            self.display_plan(tamp_problem, retime_plan(plan))


if __name__ == '__main__':
    tamp_planner = TAMPPlanner()
    tamp_planner.execute()
