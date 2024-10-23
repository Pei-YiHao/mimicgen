import sys, os
import numpy as np
import json
from typing import Sequence
import matplotlib.pyplot as plt
import base64
import h5py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def visualize_epoch(predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str, instruction: str):
    ACTION_DIM_LABELS = ["x", "y", "z", "yaw", "pitch", "roll", "grasp"]

    img_strip = np.concatenate(np.array(images[::20]), axis=1)
    # set up plt figure
    figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # plot actions
    pred_actions = np.array(predicted_raw_actions)
    for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
        # actions have batch, horizon, dim, in this example we just take the first action for simplicity
        axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
        axs[action_label].set_title(action_label)
        axs[action_label].set_xlabel("Time in one episode")

    # Show the concatenated images
    axs["image"].imshow(img_strip)
    axs["image"].set_xlabel("Time in one episode (subsampled)")

    # Add the instruction text to the plot (above the images)
    fig.text(0.5, 0.95, instruction, ha='center', fontsize=16, color='black')

    plt.legend()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
    
"""data = json.load(open("/Users/msv.sat/Desktop/mycorobot/data/data.json"))
images = []
for image in data["images"]:
    image = image["__numpy__"]
    image = base64.b64decode(image)
    image = np.frombuffer(image, dtype=np.uint8)
    image = image.reshape(256, 256, 3)
    images.append(image)
images = np.array(images)
actions = []
for action in data["actions"]:
    action = action["__numpy__"]
    action = base64.b64decode(action)
    action = np.frombuffer(action, dtype=np.float64)
    actions.append(action)
actions = [action * [560.,560.,560.,90.,90.,90.,1.] for action in actions]
actions = np.array(actions)
actions = actions[:25,...]
images = images[:25,...]
print(actions.shape,images.shape)
visualize_epoch(actions , images, "/Users/msv.sat/Desktop/mycorobot/data/pen.png")
x,y,z,yaw,pitch,roll,grasp = np.sum(actions[:,0]),np.sum(actions[:,1]),np.sum(actions[:,2]),np.sum(actions[:,3]),np.sum(actions[:,4]),np.sum(actions[:,5]),np.sum(actions[:,6])
print("x: ",x,"y: ",y,"z: ",z,"yaw: ",yaw,"pitch: ",pitch,"roll: ",roll,"grasp: ",grasp)"""

input_path = "/Volumes/gsl/mimicgen/core_datasets"
task = "coffee/demo_src_coffee_task_O1"
output_path = "/Volumes/gsl/mimicgen/demos_visualized"
with h5py.File(os.path.join(input_path, task, "demo.hdf5"), 'r') as data:
    data = data['data']
    for key, value in data.items():
        print(key)
        print(value['instructions'][0].decode())
        action=value['actions']
        img=value['obs']['agentview_image']
        visualize_epoch(action, img, os.path.join(output_path, task, f"{key}.png"),instruction=value['instructions'][0].decode())