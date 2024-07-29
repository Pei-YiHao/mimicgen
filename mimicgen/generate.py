# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
import os
import copy
import json
import argparse
import subprocess

import mimicgen

from mimicgen.utils.file_utils import config_generator_to_script_lines
from mimicgen.scripts.generate_core_configs import make_generator, make_generators, get_task_names


DEFAULT_CACHE="/data/datasets/mimicgen"
DEFAULT_OUTPUT=DEFAULT_CACHE + '/renders'
DEFAULT_CAMERAS=["agentview", "frontview", "robot0_eye_in_hand"]


def generate(tasks=[], episodes=100, output=DEFAULT_OUTPUT, cache=DEFAULT_CACHE,
             cameras=DEFAULT_CAMERAS, camera_width=512, camera_height=512, parallel=-1, **kwargs):
    """
    Render episodes for the specified tasks with trajectories and images.
    """
    if isinstance(tasks, str):
        tasks = [tasks]
    elif not isinstance(tasks, list):
        raise TypeError(f"expected tasks to be list[str] or str (was {type(tasks)})")
    
    os.makedirs(cache, exist_ok=True)
    
    task_configs = make_generators(
        src_data_dir=os.path.join(cache, 'source'), 
        output_folder=output, return_configs=True
    )
    
    datasets = {}
    generators = []
    
    for task in tasks:
        for task_config in task_configs:
            if task in task_config['tasks']:
                break
                
        dataset_name = task_config['dataset_name']
        
        if dataset_name not in datasets:
            datasets[dataset_name] = copy.deepcopy(task_config)
            datasets[dataset_name]['tasks'] = []
            datasets[dataset_name]['task_names'] = []
            
        datasets[dataset_name]['tasks'].append(task)
        datasets[dataset_name]['task_names'].append(task.split('_')[-1])
        
    download(datasets, cache=cache, **kwargs)
    
    for dataset_name, dataset in datasets.items():
        dataset_config = os.path.join(mimicgen.__path__[0], "exps/templates/robosuite", dataset_name + ".json")
        generators.append(make_generator(
            dataset_config, dataset, num_traj=episodes, 
            camera_names=cameras, camera_width=camera_width, camera_height=camera_height
        ))
        
    gen_configs, gen_commands = config_generator_to_script_lines(
        generators, config_dir=os.path.join(output, 'config')
    )
    
    for i, line in enumerate(gen_commands):
        line = line.strip().replace("train.py", "-m mimicgen.scripts.generate_dataset").replace("python", "python3")
        
        if parallel is not None and parallel != 0:
            line = line.replace('generate_dataset', 'generate_parallel')
            line += f" --workers={parallel}"
            
        line += " --auto-remove-exp"
        gen_commands[i] = line

    print("\nGeneration Configs:\n")
    print(json.dumps(gen_configs, indent=2))
    print("\nGeneration Commands:\n")
    print(json.dumps(gen_commands, indent=2))
        
    for cmd in gen_commands:
        run(cmd, **kwargs)
        
    print(f"\nFinished generation of {episodes} episodes for {tasks}")
        
        
def download(datasets={}, dataset_type='source', cache=DEFAULT_CACHE, **kwargs):
    """
    Download and prepare the source teleop datasets used for interpolation.
    """
    for dataset_name, dataset in datasets.items():       
        dataset_path = os.path.join(cache, dataset_type, dataset_name + '.hdf5')
        
        if os.path.isfile(dataset_path):
            print(f"Found dataset {dataset_name} already downloaded at {dataset_path}")
            continue
        
        print(f"\nDownloading dataset {dataset_name} for tasks {dataset['tasks']}")
        
        cmd = "python3 -m mimicgen.scripts.download_datasets "
        cmd += f"--download_dir {cache} --dataset_type {dataset_type} "
        cmd += f"--tasks {dataset_name} "
        
        run(cmd, **kwargs)

        cmd = "python3 -m mimicgen.scripts.prepare_src_dataset "
        cmd += f"--dataset {dataset_path} --env_interface_type robosuite "
        cmd += f"--env_interface MG_{dataset_name.title()} "

        run(cmd, **kwargs)
        
        
def run(cmd, executable='/bin/bash', shell=True, check=True, simulate=False, **kwargs):
    print(f"\nRUN {cmd}\n")
    if not simulate:
        subprocess.run(cmd, executable=executable, shell=shell, check=check, **kwargs)
    
    
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    parser.add_argument('--tasks', type=str, nargs='+', default='Stack_D0', choices=get_task_names(), help="one or more tasks to generate episodes for")
    parser.add_argument('--episodes', type=int, default=100, help="the number of episodes to generate")
    parser.add_argument('--cameras', type=str, nargs='+', default=DEFAULT_CAMERAS, help="one or more camera views to render")
    parser.add_argument('--camera-width', type=int, default=512, help="the width (in pixels) of the camera")
    parser.add_argument('--camera-height', type=int, default=512, help="the height (in pixels) of the camera")

    parser.add_argument('--cache', type=str, default=DEFAULT_CACHE, help="location where mimicgen datasets are stored")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help="output directory of the generated dataset (in HDF5 format)")
    parser.add_argument('--parallel', type=int, default=None, help="the number of workers to run in parallel (-1 for all CPU cores, 0 for single-threaded)")
    parser.add_argument('--simulate', action='store_true', help="just print the commands, but don't actually run them")
    
    args = parser.parse_args()
    print(args)
    
    generate(**vars(args))


if __name__ == "__main__":
    main()
    
