# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from collections import OrderedDict
import numpy as np
import random

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjmod import TextureModder
from robosuite.environments.manipulation.stack import Stack

from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class Stack_D0(Stack, SingleArmEnv_MG):
    """
    Augment robosuite stack task for mimicgen.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        self.motion_threshold = kwargs.get('motion_threshold', 0.1)
        
        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 2
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def reward(self, action=None):
        return Stack.reward(self, action=action)

    def staged_rewards(self):
        r_reach, r_lift, r_stack = super().staged_rewards()
       
        if r_stack > 0:
            xvelp = self.sim.data.get_body_xvelp(self.cubeA.root_body)
            xvelr = self.sim.data.get_body_xvelr(self.cubeA.root_body)
            if any(np.abs(xvelp) > self.motion_threshold): #or any(np.abs(xvelr) > self.motion_threshold):
                print(f'mimicgen {self.__class__.__name__} - muting reward', r_stack, 'CUBE_A xvelp', xvelp, 'xvelr', xvelr)
                r_stack = 0.0
                
        return r_reach, r_lift, r_stack
        
    def _check_lifted(self, body_id, margin=0.04):
        # lifting is successful when the cube is above the table top by a margin
        body_pos = self.sim.data.body_xpos[body_id]
        body_height = body_pos[2]
        table_height = self.table_offset[2]
        body_lifted = body_height > table_height + margin
        return body_lifted

    def _check_cubeA_lifted(self):
        return self._check_lifted(self.cubeA_body_id, margin=0.04)

    def _check_cubeA_stacked(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        cubeA_lifted = self._check_cubeA_lifted()
        cubeA_touching_cubeB = self.check_contact(self.cubeA, self.cubeB)
        return (not grasping_cubeA) and cubeA_lifted and cubeA_touching_cubeB

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.006, 0.006, 0.008],
            size_max=[0.006, 0.006, 0.008],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.015, 0.015, 0.015],
            size_max=[0.015, 0.015, 0.015],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        cubes = [self.cubeA, self.cubeB]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return { 
            k : dict(
                x=(-0.04, 0.04),
                y=(-0.04, 0.04),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((-0.4, 0, 0.95)),
            )
            for k in ["cubeA", "cubeB"]
        }


class Stack_D1(Stack_D0):
    """
    Much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        camera_pos = ["0.09 0 1.15", "0.5846 0.3977 0.3977 0.5846"]
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(camera_pos[0]),
            quat=string_to_array(camera_pos[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(camera_pos[0]),
            quat=string_to_array(camera_pos[1]),
        )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-0.1, 0.1),
                y=(-0.15, 0.15),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((-0.4, 0, 0.95)),
            )
            for k in ["cubeA", "cubeB"]
        }
        
        
class Stack_D2(Stack_D1):
    """
    Domain randomization on the block colors, in addition to wider camera view and initialization bounds like Stack_D1.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cube_keys = ['A', 'B']
        self.cube_names = [f"cube{x}_g0" for x in self.cube_keys]
        self.cube_colors = ['red', 'green']
        
        self.colors = {
            'red': np.array([255,0,0]),
            'orange': np.array([255,128,0]),
            'yellow': np.array([255,255,0]),
            'green': np.array([128,255,0]),
            'blue': np.array([0,128,255]),
            'purple': np.array([128,0,255]),
        }
        
        self.instruction_desc = {
            'PICK': [
                'stack',
                'place',
                'put',
            ],
            
            'PLACE': [
                'on the',
                'on top of the'
            ],
            
            'NOUN_A': [
                'block',
                'cube',
            ],
            
            'NOUN_B': [
                'block',
                'cube',
                'one',
            ],

            'A': [
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'small',
                'smaller',
                'little',
            ],
            
            'B': [
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'big',
                'larger',
                'other',
            ],
        }

        self.instruction = "put the red block on top of the green block"
        self.instruction_template = "$PICK the $A $NOUN_A $PLACE $B $NOUN_B"
    
    def randomize_colors(self):
        mod = TextureModder(self.sim, geom_names=self.cube_names + [x + '_vis' for x in self.cube_names])
        self.cube_colors = random.sample(self.colors.keys(), 2)
        for i, cube_name in enumerate(self.cube_names):
            mod.set_geom_rgb(cube_name, (self.colors[self.cube_colors[i]]))
            mod.set_rgb(cube_name + '_vis', self.colors[self.cube_colors[i]])
        #mod.set_rgb('table_visual', (200,200,200))       
            
    def randomize_instruction(self, store=True):
        instruction = self.instruction_template
        instruction_desc = self.randomize_descriptor(self.instruction_desc)
            
        for key, value in instruction_desc.items():
            instruction = instruction.replace(f"${key}", value)

        if store:
            self.instruction = instruction
            
        return instruction
    
    def randomize_descriptor(self, desc):
        if isinstance(desc, dict):
            return {k: self.randomize_descriptor(v) for k,v in self.instruction_desc.items()}
        elif not isinstance(desc, list):
            raise TypeError(f"expected var to be dict or list (was {type(var)})")
            
        val = desc[random.randrange(len(desc))]
        
        for i, key in enumerate(self.cube_keys):
            val = val.replace(f"COLOR_{key}", self.cube_colors[i])

        return val

    def reset(self):
        obs = super().reset()
        #print(self.sim.model.geom_names)
        self.randomize_colors()
        self.randomize_instruction()
        #print(f"\nInstruction: {self.instruction}")
        return self._get_observations(force_update=True)


class Stack_D3(Stack_D2):
    """
    Randomized colors every episode, and randomized language instructions every frame
    """
    def step(self, action):
        self.randomize_instruction()
        return super().step(action)
        

class Stack_D4(Stack_D3):
    """
    Randomized colors and language instructions every frame
    """
    def step(self, action):
        self.randomize_colors()
        return super().step(action)
        
                       
class StackThree(Stack_D0):
    """
    Stack three cubes instead of two.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 3
        for k in ["x", "y", "z_rot", "reference"]:
            assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds["cubeB"][k]))
            assert np.array_equal(np.array(bounds["cubeB"][k]), np.array(bounds["cubeC"][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def reward(self, action=None):
        """
        We only return sparse rewards here.
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _check_cubeC_lifted(self):
        # cube C needs to be higher than A
        return self._check_lifted(self.cubeC_body_id, margin=0.08)

    def _check_cubeC_stacked(self):
        grasping_cubeC = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeC)
        cubeC_lifted = self._check_cubeC_lifted()
        cubeC_touching_cubeA = self.check_contact(self.cubeC, self.cubeA)
        return (not grasping_cubeC) and cubeC_lifted and cubeC_touching_cubeA

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.
        Returns:
            3-tuple:
                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # Stacking successful when A is on top of B and C is on top of A.
        # This means both A and C are lifted, not grasped by robot, and we have contact
        # between (A, B) and (A, C).

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_reach = 0.
        r_lift = 0.
        r_stack = 0.
        if self._check_cubeA_stacked() and self._check_cubeC_stacked():
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_arena(self):
        """
        Allow subclasses to easily override arena settings.
        """

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        return mujoco_arena

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = self._load_arena()

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.015, 0.015, 0.015],
            size_max=[0.015, 0.015, 0.015],
            rgba=[1, 0, 0, 1],
            material=bluewood,
        )
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _setup_references(self):
        """
        Add reference for cube C
        """
        super()._setup_references()

        # Additional object references from this env
        self.cubeC_body_id = self.sim.model.body_name2id(self.cubeC.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cubeC_body_id])

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cubeC_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache[f"{pf}eef_pos"] if \
                    "cubeC_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeA_to_cubeC(obs_cache):
                return obs_cache["cubeC_pos"] - obs_cache["cubeA_pos"] if \
                    "cubeA_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def cubeB_to_cubeC(obs_cache):
                return obs_cache["cubeB_pos"] - obs_cache["cubeC_pos"] if \
                    "cubeB_pos" in obs_cache and "cubeC_pos" in obs_cache else np.zeros(3)

            sensors = [cubeC_pos, cubeC_quat, gripper_to_cubeC, cubeA_to_cubeC, cubeB_to_cubeC]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return { 
            k : dict(
                x=(-0.10, 0.10),
                y=(-0.10, 0.10),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }


class StackThree_D0(StackThree):
    """Rename base class for convenience."""
    pass


class StackThree_D1(StackThree_D0):
    """
    Less z-rotation (for easier datagen) and much wider initialization bounds.
    """
    def _load_arena(self):
        """
        Make default camera have full view of tabletop to account for larger init bounds.
        """
        mujoco_arena = super()._load_arena()

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        for view in ['agentview']: #, 'frontview']:
            old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": view}, return_first=True)
            old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
            old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": f"{view}_full"}, return_first=True)
            old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
            mujoco_arena.set_camera(
                camera_name=view,
                pos=string_to_array(old_agentview_full_camera_pose[0]),
                quat=string_to_array(old_agentview_full_camera_pose[1]),
            )
            mujoco_arena.set_camera(
                camera_name=f"{view}_full",
                pos=string_to_array(old_agentview_camera_pose[0]),
                quat=string_to_array(old_agentview_camera_pose[1]),
            )

        return mujoco_arena

    def _get_initial_placement_bounds(self):
        max_dim = 0.20
        return { 
            k : dict(
                x=(-max_dim, max_dim),
                y=(-max_dim, max_dim),
                z_rot=(0., 2. * np.pi),
                # NOTE: hardcoded @self.table_offset since this might be called in init function
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC"]
        }

class StackThree_D2(StackThree_D1):
    """
    在每个回合随机化颜色，并在回合开始时随机化语言指令
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cube_keys = ['A', 'B', 'C']  # 包含三个方块
        self.cube_names = [f"cube{x}_g0" for x in self.cube_keys]
        self.cube_colors = ['red', 'green', 'blue']  # 默认颜色为红、绿、蓝
        
        self.colors = {
            'red': (255, 0, 0),
            'orange': (255, 128, 0),
            'yellow': (255, 255, 0),
            'green': (128, 255, 0),
            'blue': (0, 128, 255),
            'purple': (128, 0, 255),
            'black': (12, 12, 12),
            'gray': (128, 128, 128),
            'dark gray': (64, 64, 64),
            'pink': (255, 153, 255),
            'brown': (153,76,0),
        }
        
        # 新增三块方块的描述
        self.instruction_desc = {
            'PICK': [
                'stack',
                'place',
                'put',
            ],
            'PLACE': [
                'on the',
                'on top of the'
            ],
            'NOUN_A': [
                'block',
                'cube',
            ],
            'NOUN_B': [
                'block',
                'cube',
                'one',
            ],
            'A': [
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'COLOR_A',
                'medium',
            ],
            'B': [
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'COLOR_B',
                'biggest'
            ],
            'C': [
                'COLOR_C',
                'COLOR_C',
                'COLOR_C',
                'COLOR_C',
                'COLOR_C',
                'smallest'
            ]
        }

        # 默认指令
        self.instruction = "put the red block on top of the green block, then place the blue block on top"
        self.instruction_template = "$PICK the $A $NOUN_A $PLACE $B $NOUN_B, then $PICK the $C $NOUN_A $PLACE $A $NOUN_B"
        self.instruction_templates = [
            "$PICK the $A $NOUN_A $PLACE $B $NOUN_B, then $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "1. $PICK the $A $NOUN_A $PLACE $B $NOUN_B. 2. $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "$PICK the $A $NOUN_A $PLACE $B $NOUN_B, then $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "1. $PICK the $A $NOUN_A $PLACE $B $NOUN_B. 2. $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "1. $PICK the $A $NOUN_A $PLACE $B $NOUN_B. 2. $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "1. $PICK the $A $NOUN_A $PLACE $B $NOUN_B. 2. $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
            "firstly $PICK the $A $NOUN_A $PLACE $B $NOUN_B, after that $PICK the $C $NOUN_A $PLACE $A $NOUN_B",
        ]
        
    def randomize_colors(self):
        # 随机选择三种颜色
        mod = TextureModder(self.sim, geom_names=self.cube_names + [x + '_vis' for x in self.cube_names])
        self.cube_colors = random.sample(self.colors.keys(), 3)

        for i, cube_name in enumerate(self.cube_names):
            mod.set_geom_rgb(cube_name, self.colors[self.cube_colors[i]])
            mod.set_rgb(cube_name + '_vis', self.colors[self.cube_colors[i]])
        

    def randomize_instruction(self, store=True):
        instruction = random.sample(self.instruction_templates, 1)[0]
        instruction_desc = self.randomize_descriptor(self.instruction_desc)
            
        for key, value in instruction_desc.items():
            instruction = instruction.replace(f"${key}", value)

        if store:
            self.instruction = instruction
            
        return instruction
    
    def randomize_descriptor(self, desc):
        if isinstance(desc, dict):
            return {k: self.randomize_descriptor(v) for k,v in desc.items()}
        elif not isinstance(desc, list):
            raise TypeError(f"expected var to be dict or list (was {type(desc)})")
            
        val = desc[random.randrange(len(desc))]
        
        for i, key in enumerate(self.cube_keys):
            val = val.replace(f"COLOR_{key}", self.cube_colors[i])

        return val

    def reset(self):
        obs = super().reset()
        self.randomize_colors()
        self.randomize_instruction()
        return self._get_observations(force_update=True)

class StackThree_D3(StackThree_D2):
    """
    在每个回合随机化颜色，并在每帧随机化语言指令
    """
    def step(self, action):
        # 每帧随机化语言指令
        self.randomize_instruction()
        return super().step(action)


class StackFour(Stack_D0):
    """
    Stack four cubes instead of two.
    A is placed on B, C is placed on A, and D is placed on C.
    """
    def __init__(self, **kwargs):
        assert "placement_initializer" not in kwargs, "this class defines its own placement initializer"

        bounds = self._get_initial_placement_bounds()

        # ensure cube symmetry
        assert len(bounds) == 4
        for k in ["x", "y", "z_rot", "reference"]:
            for cube_key in ['cubeA', 'cubeB', 'cubeC', 'cubeD']:
                assert np.array_equal(np.array(bounds["cubeA"][k]), np.array(bounds[cube_key][k]))

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=bounds["cubeA"]["x"],
            y_range=bounds["cubeA"]["y"],
            rotation=bounds["cubeA"]["z_rot"],
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=bounds["cubeA"]["reference"],
            z_offset=0.01,
        )

        Stack_D0.__init__(self, placement_initializer=placement_initializer, **kwargs)

    def _check_cubeD_stacked(self):
        grasping_cubeD = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeD)
        cubeD_lifted = self._check_lifted(self.cubeD_body_id, margin=0.08)
        cubeD_touching_cubeC = self.check_contact(self.cubeD, self.cubeC)
        return (not grasping_cubeD) and cubeD_lifted and cubeD_touching_cubeC

    def staged_rewards(self):
        r_reach, r_lift, r_stack = super().staged_rewards()

        # Stacking is successful when A is on B, C is on A, and D is on C.
        if self._check_cubeA_stacked() and self._check_cubeC_stacked() and self._check_cubeD_stacked():
            r_stack = 1.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        yellowwood = CustomMaterial(
            texture="WoodYellow",
            tex_name="yellowwood",
            mat_name="yellowwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.015, 0.015, 0.015],
            size_max=[0.015, 0.015, 0.015],
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )
        self.cubeD = BoxObject(
            name="cubeD",
            size_min=[0.01, 0.01, 0.01],
            size_max=[0.01, 0.01, 0.01],
            rgba=[1, 1, 0, 1],
            material=yellowwood,
        )

        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,
                x_range=[-0.10, 0.10],
                y_range=[-0.10, 0.10],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=self._load_arena(),
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects.
        Should return a dictionary with bounds for four cubes.
        """
        return {
            k: dict(
                x=(-0.10, 0.10),
                y=(-0.10, 0.10),
                z_rot=(0., 2. * np.pi),
                reference=np.array((0, 0, 0.8)),
            )
            for k in ["cubeA", "cubeB", "cubeC", "cubeD"]
        }
