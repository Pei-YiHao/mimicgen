#!/bin/bash

# Store commands used to prepare source datasets for each environment.
DATA=/data/datasets/mimicgen/source

# coffee
python3 prepare_src_dataset.py \
    --dataset $DATA/coffee.hdf5 \
    --env_interface MG_Coffee \
    --env_interface_type robosuite

# coffee_preparation
python3 prepare_src_dataset.py \
    --dataset $DATA/coffee_preparation.hdf5 \
    --env_interface MG_CoffeePreparation \
    --env_interface_type robosuite

# hammer_cleanup
python3 prepare_src_dataset.py \
    --dataset $DATA/hammer_cleanup.hdf5 \
    --env_interface MG_HammerCleanup \
    --env_interface_type robosuite

# kitchen
python3 prepare_src_dataset.py \
    --dataset $DATA/kitchen.hdf5 \
    --env_interface MG_Kitchen \
    --env_interface_type robosuite

# mug_cleanup
python3 prepare_src_dataset.py \
    --dataset $DATA/mug_cleanup.hdf5 \
    --env_interface MG_MugCleanup \
    --env_interface_type robosuite

# nut_assembly
python3 prepare_src_dataset.py \
    --dataset $DATA/nut_assembly.hdf5 \
    --env_interface MG_NutAssembly \
    --env_interface_type robosuite

# pick_place
python3 prepare_src_dataset.py \
    --dataset $DATA/pick_place.hdf5 \
    --env_interface MG_PickPlace \
    --env_interface_type robosuite

# square
python3 prepare_src_dataset.py \
    --dataset $DATA/source/square.hdf5 \
    --env_interface MG_Square \
    --env_interface_type robosuite

# stack
python3 prepare_src_dataset.py \
    --dataset $DATA/source/stack.hdf5 \
    --env_interface MG_Stack \
    --env_interface_type robosuite

# stack_three
python3 prepare_src_dataset.py \
    --dataset $DATA/stack_three.hdf5 \
    --env_interface MG_StackThree \
    --env_interface_type robosuite

# threading
python3 prepare_src_dataset.py \
    --dataset $DATA/threading.hdf5 \
    --env_interface MG_Threading \
    --env_interface_type robosuite

# three_piece_assembly
python3 prepare_src_dataset.py \
    --dataset $DATA/three_piece_assembly.hdf5 \
    --env_interface MG_ThreePieceAssembly \
    --env_interface_type robosuite
