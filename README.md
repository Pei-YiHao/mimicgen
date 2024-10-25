# Mycobot-MimicGen

Based on [Robosuite](https://github.com/ARISE-Initiative/robosuite) and [MimicGen](https://github.com/NVlabs/mimicgen). For Robosuite-related content, refer to: https://github.com/Pei-YiHao/robosuite-mycobot. This project extends MimicGen with more complex instruction generation. Additional domain randomization was added, including color, size, instructions, spatial position, texture*, and lighting*. It also includes support for the Mycobot280 6-DOF collaborative robotic arm, tested with a 40.5% success rate in the Stack D4 task using Mycobot280 + parallel gripper.

### References:

- Robosuite: https://github.com/dusty-nv/robosuite
- MimicGen: [MimicGen Paper](https://arxiv.org/pdf/2310.17596) and https://github.com/dusty-nv/mimicgen
- Mycobot-Robosuite: https://github.com/Pei-YiHao/robosuite-mycobot

### Installation:

To install MimicGen:
```bash
git clone https://github.com/Pei-YiHao/mimicgen
pip install -e .
