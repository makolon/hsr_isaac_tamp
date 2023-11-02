import os
import torch
import numpy as np
from typing import Optional
from pxr import Gf
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.physx.scripts import utils
from pxr import UsdGeom
from hsr_rl.utils.files import get_usd_path
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats

def spawn_obstacle(name, prim_path, device):
    object_translation = torch.tensor([1.7, 0.0, 0.055], device=device)
    object_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Spawn object model from usd path
    object_usd_path = os.path.join(get_usd_path(), 'gearbox', name, name+'.usd')

    add_reference_to_stage(object_usd_path, prim_path+"/obstacle/"+name)
    obst_prim = XFormPrim(
        prim_path=prim_path+"/obstacle/"+name,
        translation=object_translation,
        orientation=object_orientation
    )

    return obst_prim

def spawn_dynamic_object(name, prim_path, object_translation, object_orientation):
    # Spawn object model from usd path
    object_usd_path = os.path.join(get_usd_path(), 'gearbox_dynamic', name, name+'.usd')

    add_reference_to_stage(object_usd_path, prim_path+"/"+name)
    obj_prim = XFormPrim(
        prim_path=prim_path+"/"+name,
        translation=object_translation,
        orientation=object_orientation
    )

    return obj_prim

def spawn_static_object(name, prim_path, object_translation, object_orientation):
    # Spawn object model from usd path
    object_usd_path = os.path.join(get_usd_path(), 'gearbox_static', name, name+'.usd')

    add_reference_to_stage(object_usd_path, prim_path+"/"+name)
    obj_prim = XFormPrim(
        prim_path=prim_path+"/"+name,
        translation=object_translation,
        orientation=object_orientation
    )

    return obj_prim

def setup_gearbox_scene(env_ids, indices, obstacles, obstacles_state, grasp_objs, grasp_objs_state, device):
    # Place all grasp objects on the tabular obstacle (without overlaps)
    for idx, _ in enumerate(obstacles):
        obst_position = obstacles_state[idx][env_ids, :3]
        obst_orientation = obstacles_state[idx][env_ids, 3:]
        obstacles[idx].set_world_poses(positions=obst_position,
                                        orientations=obst_orientation,
                                        indices=indices)
    for idx, _ in enumerate(grasp_objs):
        grasp_obj_position = grasp_objs_state[idx][env_ids, :3]
        grsap_obj_orientation = grasp_objs_state[idx][env_ids, 3:]
        grasp_objs[idx].set_world_poses(positions=grasp_obj_position,
                                        orientations=grsap_obj_orientation,
                                        indices=indices)

    # Pick one object to be the grasp object and compute its grasp:
    goal_obj_index = np.random.randint(len(grasp_objs))

    return grasp_objs[goal_obj_index]

class DynamicObject(RigidPrim, GeometryPrim):
    """Creates and adds a prim to stage from USD reference path, and wraps the prim with RigidPrim and GeometryPrim to 
       provide access to APIs for rigid body attributes, physics materials and collisions. Please note that this class 
       assumes the object has only a single mesh prim defining its geometry.

    Args:
        usd_path (str): USD reference path the Prim refers to.
        prim_path (str): prim path of the Prim to encapsulate or create.
        mesh_path (str): prim path of the underlying mesh Prim.
        name (str, optional): shortname to be used as a key by Scene class. Note: needs to be unique if the object is 
                              added to the Scene. Defaults to "dynamic_object".
        position (Optional[np.ndarray], optional): position in the world frame of the prim. Shape is (3, ). Defaults to 
                                                   None, which means left unchanged.
        translation (Optional[np.ndarray], optional): translation in the local frame of the prim (with respect to its 
                                                      parent prim). Shape is (3, ). Defaults to None, which means left 
                                                      unchanged.
        orientation (Optional[np.ndarray], optional): quaternion orientation in the world/local frame of the prim
                                                      (depends if translation or position is specified). Quaternion is 
                                                      scalar-first (w, x, y, z). Shape is (4, ). Defaults to None, which 
                                                      means left unchanged.
        scale (Optional[np.ndarray], optional): local scale to be applied to the prim's dimensions. Shape is (3, ).
                                                Defaults to None, which means left unchanged.
        visible (bool, optional): set to false for an invisible prim in the stage while rendering. Defaults to True.
        mass (Optional[float], optional): mass in kg. Defaults to None.
        linear_velocity (Optional[np.ndarray], optional): linear velocity in the world frame. Defaults to None.
        angular_velocity (Optional[np.ndarray], optional): angular velocity in the world frame. Defaults to None.
    """

    def __init__(
        self,
        usd_path: str,
        prim_path: str,
        mesh_path: str,
        name: str = "dynamic_object",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: bool = True,
        mass: Optional[float] = None,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ) -> None:

        if is_prim_path_valid(mesh_path):
            prim = get_prim_at_path(mesh_path)
            if not prim.IsA(UsdGeom.Mesh):
                raise Exception("The prim at path {} cannot be parsed as a Mesh object".format(mesh_path))

        self.usd_path = usd_path

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        GeometryPrim.__init__(
            self,
            prim_path=mesh_path,
            name=name,
            translation=translation,
            orientation=orientation,
            visible=visible,
            collision=True,
        )

        self.set_collision_approximation("convexHull")

        RigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )