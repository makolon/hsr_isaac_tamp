from collections import namedtuple

import numpy as np
import string
import random

GROUND_NAME = 'gray'
BLOCK_WIDTH = 2
BLOCK_HEIGHT = BLOCK_WIDTH
GROUND_Y = 0.

SUCTION_HEIGHT = 1.
GRASP = -np.array([0, BLOCK_HEIGHT + SUCTION_HEIGHT/2]) # TODO: side grasps
CARRY_Y = 2*BLOCK_WIDTH+SUCTION_HEIGHT
APPROACH = -np.array([0, CARRY_Y]) - GRASP

MOVE_COST = 10.
COST_PER_DIST = 1.
DISTANCE_PER_TIME = 4.0


def get_block_box(b, p=np.zeros(2)):
    extent = np.array([BLOCK_WIDTH, BLOCK_HEIGHT]) # TODO: vary per block
    lower = p - extent/2.
    upper = p + extent/2.
    return lower, upper


def boxes_overlap(box1, box2):
    lower1, upper1 = box1
    lower2, upper2 = box2
    return np.less_equal(lower1, upper2).all() and \
           np.less_equal(lower2, upper1).all()


def interval_contains(i1, i2):
    """
    :param i1: The container interval
    :param i2: The possibly contained interval
    :return:
    """
    return (i1[0] <= i2[0]) and (i2[1] <= i1[1])


def interval_overlap(i1, i2):
    return (i2[0] <= i1[1]) and (i1[0] <= i2[1])


def get_block_interval(b, p):
    lower, upper = get_block_box(b, p)
    return lower[0], upper[0]


def get_block_interval_margin(b, p):
    collision_margin = 0.1
    lower, upper = get_block_box(b, p)
    return lower[0]-collision_margin, upper[0]+collision_margin


def sample_region(b, region):
    x1, x2 = np.array(region, dtype=float) - get_block_interval(b, np.zeros(2))
    if x2 < x1:
        return None
    x = np.random.uniform(x1, x2)
    return np.array([x, 0])


def rejection_sample_region(b, region, placed={}, max_attempts=10):
    for _ in range(max_attempts):
        p = sample_region(b, region)
        if p is None:
            break
        if not any(collision_test(b, p, b2, p2) for b2, p2 in placed.items()):
            return p
    return None


def rejection_sample_placed(block_poses={}, block_regions={}, regions={}, max_attempts=10):
    assert(not set(block_poses.keys()) & set(block_regions.keys()))
    for _ in range(max_attempts):
        placed = block_poses.copy()
        remaining = list(block_regions.items())
        random.shuffle(remaining)
        for b, r in remaining:
            p = rejection_sample_region(b, regions[r], placed)
            if p is None:
                break
            placed[b] = p
        else:
            return placed
    return None


def sample_center_region(b, region):
    x1, x2 = np.array(region, dtype=float) - get_block_interval(b, np.zeros(2))
    if x2 < x1:
        return None
    x = (x1+x2)/2.
    return np.array([x, 0])


################################################## Streams

def collision_test(b1, p1, b2, p2):
    if b1 == b2:
        return False
    return interval_overlap(get_block_interval(b1, p1),
                            get_block_interval(b2, p2))


def collision_test(b1, p1, b2, p2):
    if b1 == b2:
        return False
    return interval_overlap(get_block_interval_margin(b1, p1),
                            get_block_interval_margin(b2, p2))


def stack_test(bu, pu, bl, pl):
    if interval_contains(get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2., get_block_interval(bu, pu)):
        return (pu[1] > pl[1]) and (pu[1] <= pl[1]+BLOCK_HEIGHT)
    else:
        return False


def stack_free_test(b1, p1, b2, p2):
    if interval_overlap(get_block_interval(b1, p1), get_block_interval(b2, p2)):
        return (p1[1] > p2[1])
    else:
        return False


def str_eq(s1, s2, ignore_case=True):
    if ignore_case:
        s1 = s1.lower()
        s2 = s2.lower()
    return s1 == s2


def interpolate(waypoints, velocity=1):
    import scipy
    differences = [0.] + [np.linalg.norm(q2 - q1) for q1, q2 in zip(waypoints[:-1], waypoints[1:])]
    times = np.cumsum(differences) / velocity
    return scipy.interpolate.interp1d(times, waypoints, kind='linear')


def plan_motion(q1, q2, fluents=[]):
    x1, y1 = q1
    x2, y2 = q2
    t = [q1, np.array([x1, CARRY_Y]), np.array([x2, CARRY_Y]), q2]

    grasp = None
    placed = {}
    for fluent in fluents:
        predicate, args = fluent[0], fluent[1:]
        if str_eq(predicate, 'AtGrasp'):
            assert grasp is None
            r, b, g = args
            grasp = g
        elif str_eq(predicate, 'AtPose'):
            b, p = args
            assert b not in placed
            placed[b] = p
        elif str_eq(predicate, 'AtConf'):
            continue
        else:
            raise NotImplementedError(predicate)

    if grasp is None:
        return (t,)
    for q in t:
        p1 = forward_kin(q, grasp)
        box1 = get_block_box(None, p1)
        for b2, p2 in placed.items():
            box2 = get_block_box(b2, p2)
            if boxes_overlap(box1, box2):
                return None
    return (t,)


def forward_kin(q, g):
    p = q + g
    return p


def inverse_kin(p, g):
    q = p - g
    return q


def approach_kin(q):
    a = q - APPROACH
    return a


def distance_fn(q1, q2):
    ord = 1  # 1 | 2
    return MOVE_COST + COST_PER_DIST*np.linalg.norm(q2 - q1, ord=ord)


def pick_distance_fn(q1, b, q2):
    ord = 1  # 1 | 2
    return MOVE_COST + COST_PER_DIST*np.linalg.norm(q2 - q1, ord=ord)


def duration_fn(traj):
    distance = sum(np.linalg.norm(q2 - q1) for q1, q2 in zip(traj, traj[1:]))
    return distance / DISTANCE_PER_TIME


def inverse_kin_fn(b, p, g):
    q = inverse_kin(p, g)
    #return (q,)
    a = approach_kin(q)
    return (a,)


def stack_inverse_kin_fn(b, p, g):
    q = p - g
    a = q - APPROACH - np.array([0, p[1]])
    return (a,)


def unreliable_ik_fn(*args):
    # For testing the algorithms
    while 1e-2 < np.random.random():
        yield None
    yield inverse_kin_fn(*args)


def get_region_test(regions=[]):
    def test(b, p, r):
        return interval_contains(regions[r], get_block_interval(b, p))
    return test


def get_block_region_test(regions=[]):
    def test(bu, pu, bl, pl):
        if interval_contains(get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2., get_block_interval(bu, pu)):
            return (pu[1] > pl[1]) and (pu[1] <= pl[1]+BLOCK_HEIGHT+0.1)
        else:
            return False
    return test


def get_reach_test(robot_movable_region):
    def test(r, q):
        return (robot_movable_region[r][0] <= q[0]) and (q[0] <= robot_movable_region[r][1])
    return test


def get_pose_gen(regions=[]):
    def gen_fn(b, r):
        while True:
            p = sample_region(b, regions[r])
            if p is None:
                break
            yield (p,)
    return gen_fn


def get_center_pose_gen(regions=[]):
    def gen_fn(b, r):
        while True:
            p = sample_center_region(b, regions[r])
            if p is None:
                break
            yield (p,)
    return gen_fn


def get_block_pose_gen(regions=[]):
    def get_fn(bu, bl, pl):
        while True:
            p = sample_region(bu, get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2.)
            if p is None:
                break
            p[1] = pl[1] + BLOCK_HEIGHT
            yield (p,)
    return get_fn


def get_block_center_pose_gen(regions=[]):
    def get_fn(bu, bl, pl):
        while True:
            p = sample_center_region(bu, get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2.)
            if p is None:
                break
            p[0] = pl[0]
            p[1] = pl[1] + BLOCK_HEIGHT
            yield (p,)
    return get_fn


def get_stack_pose_gen(regions=[]):
    def get_fn(bu, bl, pl):
        while True:
            p = sample_region(bu, get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2.)
            if p is None:
                break
            if stack_test(p, pl):
                yield (p,)
    return get_fn


def get_stack_center_pose_gen(regions=[]):
    def get_fn(bu, bl, pl):
        while True:
            p = sample_center_region(bu, get_block_interval(bl, pl)+np.array([-BLOCK_WIDTH,+BLOCK_WIDTH])/2.)
            if p is None:
                break
            if stack_test(p, pl):
                yield (p,)
    return get_fn


################################################## Problems

ENVIRONMENT_NAMES = [GROUND_NAME]
TAMPState = namedtuple('TAMPState', ['robot_confs', 'holding', 'block_poses'])
TAMPProblem = namedtuple('TAMPProblem', ['initial', 'regions',
                                        'goal_conf', 'goal_in', 'goal_on'])

GOAL1_NAME = 'red'
GOAL2_NAME = 'orange'
STOVE_NAME = 'stove'
TABLE_NAME = 'table'

INITIAL_CONF = np.array([-5, CARRY_Y + 1])
GOAL_CONF = INITIAL_CONF

REGIONS = {
    GROUND_NAME: (-10, 10),
    GOAL1_NAME: (-2.25, -0.25),
    GOAL2_NAME: (0.25, 2.25),
}

def make_blocks(num):
    return [string.ascii_uppercase[i] for i in range(num)]


def gearbox(n_blocks=4, n_robots=1, deterministic=True):
    confs = [INITIAL_CONF, np.array([-1, 1])*INITIAL_CONF]
    robots = ['r{}'.format(x) for x in range(n_robots)]
    initial_confs = dict(zip(robots, confs))

    blocks = make_blocks(n_blocks)
    if deterministic:
        lower, upper = REGIONS[GROUND_NAME]
        poses = [np.array([-7.5, 0]), np.array([-5.0, 0]), np.array([5.0, 0]), np.array([7.5, 0])]
        poses.extend(np.array([lower + BLOCK_WIDTH/2 + (BLOCK_WIDTH + 1) * x, 0])
                     for x in range(n_blocks-len(poses)))
        block_poses = dict(zip(blocks, poses))
    else:
        block_regions = {blocks[0]: GROUND_NAME}
        block_regions.update({b: GOAL1_NAME for b in blocks[1:2]})
        block_regions.update({b: GROUND_NAME for b in blocks[2:]})
        block_poses = rejection_sample_placed(block_regions=block_regions, regions=REGIONS)

    initial = TAMPState(initial_confs, {}, block_poses)
    goal_in = {blocks[0]: GOAL1_NAME, blocks[2]: GOAL2_NAME}
    goal_on = ((blocks[0], blocks[1]), (blocks[2], blocks[3]))

    return TAMPProblem(initial, REGIONS, GOAL_CONF, goal_in, goal_on)


PROBLEMS = [
    gearbox,
]


################################################## Draw functions

def draw_robot(viewer, robot, pose, **kwargs):
    x, y = pose
    viewer.draw_robot(x, y, name=robot, **kwargs)


def draw_block(viewer, block, pose, **kwargs):
    x, y = pose
    viewer.draw_block(x, y, BLOCK_WIDTH, BLOCK_HEIGHT, name=block, **kwargs)


def draw_state(viewer, state, colors):
    # TODO: could draw the current time
    viewer.clear_state()
    #viewer.draw_environment()
    print(state)
    for robot, conf in state.robot_confs.items():
        draw_robot(viewer, robot, conf)
    for block, pose in state.block_poses.items():
        draw_block(viewer, block, pose, color=colors[block])
    for robot, holding in state.holding.items():
        block, grasp = holding
        pose = forward_kin(state.robot_confs[robot], grasp)
        draw_block(viewer, block, pose, color=colors[block])
    viewer.tk.update()


def get_random_seed():
    return np.random.get_state()[1][0]


##################################################

def apply_action(state, action):
    robot_confs, holding, block_poses = state
    # TODO: don't mutate block_poses?
    name, args = action[:2]
    if name == 'move':
        if len(args) == 4:
            robot, _, traj, _ = args
        else:
            robot, q1, q2 = args
            traj = [q1, q2]
        #traj = plan_motion(*args)[0] if len(args) == 2 else args[1]
        for conf in traj[1:]:
            robot_confs[robot] = conf
            yield TAMPState(robot_confs, holding, block_poses)
    elif name == 'pick':
        # TODO: approach and retreat trajectory
        robot, block, _, grasp, _ = args
        holding[robot] = (block, grasp)
        del block_poses[block]
        yield TAMPState(robot_confs, holding, block_poses)
    elif name == 'place':
        robot, block, pose, _, _ = args
        del holding[robot]
        block_poses[block] = pose
        yield TAMPState(robot_confs, holding, block_poses)
    else:
        raise ValueError(name)


##################################################

def prune_duplicates(traj):
    # TODO: could use the more general sparcify function
    new_traj = [traj[0]]
    for conf in traj[1:]:
        if 0 < np.linalg.norm(np.array(conf) - np.array(new_traj[-1])):
            new_traj.append(conf)
    return new_traj


def get_value_at_time(traj, fraction):
    waypoints = prune_duplicates(traj)
    if len(waypoints) == 1:
        return waypoints[0]
    distances = [0.] + [np.linalg.norm(np.array(q2) - np.array(q1))
                        for q1, q2 in zip(waypoints, waypoints[1:])]
    cum_distances = np.cumsum(distances)
    cum_fractions = np.minimum(cum_distances / cum_distances[-1], np.ones(cum_distances.shape))
    index = np.digitize(fraction, cum_fractions, right=False)
    if index == len(waypoints):
        index -= 1
    waypoint_fraction = (fraction - cum_fractions[index - 1]) / (cum_fractions[index] - cum_fractions[index - 1])
    waypoint1, waypoint2 = np.array(waypoints[index - 1]), np.array(waypoints[index])
    conf = (1 - waypoint_fraction) * waypoint1 + waypoint_fraction * waypoint2

    return conf


def update_state(state, action, t):
    robot_confs, holding, block_poses = state
    name, args, start, duration = action
    fraction = float(t) / duration
    fraction = max(0, min(fraction, 1))
    assert 0 <= fraction <= 1
    threshold = 0.5
    if name == 'move':
        robot, _, traj, _ = args
        robot_confs[robot] = get_value_at_time(traj, fraction)
    elif name == 'pick':
        robot, block, pose, grasp, conf = args[:5]
        traj = [conf, pose - grasp]
        if fraction < threshold:
            robot_confs[robot] = get_value_at_time(traj, fraction / threshold)
        else:
            holding[robot] = (block, grasp)
            block_poses.pop(block, None)
            robot_confs[robot] = get_value_at_time(traj[::-1], (fraction - threshold) / (1 - threshold))
    elif name == 'place':
        robot, block, pose, grasp, conf = args[:5]
        traj = [conf, pose - grasp]
        if fraction < threshold:
            robot_confs[robot] = get_value_at_time(traj, fraction / threshold)
        else:
            holding.pop(robot, None)
            block_poses[block] = pose
            robot_confs[robot] = get_value_at_time(traj[::-1], (fraction - threshold) / (1 - threshold))
    elif name == 'stack':
        robot, u_block, u_pose, grasp, conf, l_block, l_pose = args
        traj = [conf, u_pose - grasp]
        if fraction < threshold:
            robot_confs[robot] = get_value_at_time(traj, fraction / threshold)
        else:
            holding.pop(robot, None)
            block_poses[u_block] = u_pose
            robot_confs[robot] = get_value_at_time(traj[::-1], (fraction - threshold) / (1 - threshold))
    elif name == 'cook':
        # TODO: update the object color
        pass
    else:
        raise ValueError(name)
    return TAMPState(robot_confs, holding, block_poses)
