import os
import sys
import time
import json
import trimesh
import torch
import viser

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler


def get_link_dir(robot_name, joint_name):
    if joint_name.startswith('virtual'):
        return None

    if robot_name in ('allegro', 'allegro_mirror'):
        # allegro_mirror is D(R,O)'s own left allegro reflected through the
        # xz-plane into a right hand (scripts/mirror_urdf.py), so the model is
        # still looking at the geometry its checkpoint was trained on.
        #
        # The annotation carries over unchanged, and that is checked rather
        # than assumed: scripts/compute_link_dir.py reads [0, 0, 1] off the
        # mirrored chain for 15 of the 16 joints, because the mirror negates y
        # and this direction has no y component. joint_12.0 reads back tilted,
        # as it already does on the unmirrored hand -- upstream lumps it in
        # with the rest, and so do we.
        if joint_name in ['joint_0.0', 'joint_4.0', 'joint_8.0', 'joint_13.0']:
            return None
        link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
    elif robot_name == 'allegro_ocir':
        # Same hand as 'allegro', but OCIR's URDF, so that the model solves on
        # the exact chain the simulator replays. The two URDFs are not related
        # by any rigid transform, so nothing here can be copied from the entry
        # above -- in particular the link direction is NEGATED. Both were read
        # off the chain by scripts/compute_link_dir.py: the flexion links
        # extend along -z from their joints here and along +z there. Copying
        # [0, 0, 1] over would invert open/close silently, making the pregrasp
        # squeeze shut instead of opening.
        if joint_name in ['index_joint_0', 'middle_joint_0', 'ring_joint_0',
                          'thumb_joint_1']:
            return None
        link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)
    elif robot_name == 'barrett':
        if joint_name in ['bh_j11_joint', 'bh_j21_joint']:
            return None
        link_dir = torch.tensor([-1, 0, 0], dtype=torch.float32)
    elif robot_name == 'ezgripper':
        link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'robotiq_3finger':
        if joint_name in ['gripper_fingerB_knuckle', 'gripper_fingerC_knuckle']:
            return None
        link_dir = torch.tensor([0, 0, -1], dtype=torch.float32)
    elif robot_name == 'shadowhand':
        if joint_name in ['WRJ2', 'WRJ1']:
            return None
        if joint_name != 'THJ5':
            link_dir = torch.tensor([0, 0, 1], dtype=torch.float32)
        else:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
    elif robot_name == 'leaphand':
        if joint_name in ['13']:
            return None
        if joint_name in ['0', '4', '8']:
            link_dir = torch.tensor([1, 0, 0], dtype=torch.float32)
        elif joint_name in ['1', '5', '9', '12', '14']:
            link_dir = torch.tensor([0, 1, 0], dtype=torch.float32)
        else:
            link_dir = torch.tensor([0, -1, 0], dtype=torch.float32)
    else:
        raise NotImplementedError(f"Unknown robot name: {robot_name}!")

    return link_dir


def controller(robot_name, q_para):
    q_batch = torch.atleast_2d(q_para)

    hand = create_hand_model(robot_name, device=q_batch.device)
    joint_orders = hand.get_joint_orders()
    pk_chain = hand.pk_chain
    if q_batch.shape[-1] != len(pk_chain.get_joint_parameter_names()):
        q_batch = q_rot6d_to_q_euler(q_batch)
    status = pk_chain.forward_kinematics(q_batch)

    outer_q_batch = []
    inner_q_batch = []
    for batch_idx in range(q_batch.shape[0]):
        joint_dots = {}
        for frame_name in pk_chain.get_frame_names():
            frame = pk_chain.find_frame(frame_name)
            joint = frame.joint
            link_dir = get_link_dir(robot_name, joint.name)
            if link_dir is None:
                continue

            frame_transform = status[frame_name].get_matrix()[batch_idx]
            axis_dir = frame_transform[:3, :3] @ joint.axis
            # get_link_dir builds its tensor on the CPU while the chain has
            # been moved to the solve device, so this matmul raises as soon as
            # the hand is solved on a GPU. Upstream only ever runs the
            # controller on the CPU, so the mismatch never surfaces there.
            link_dir = frame_transform[:3, :3] @ link_dir.to(frame_transform.device)
            normal_dir = torch.cross(axis_dir, link_dir, dim=0)
            axis_origin = frame_transform[:3, 3]
            origin_dir = -axis_origin / torch.norm(axis_origin)
            joint_dots[joint.name] = torch.dot(normal_dir, origin_dir)

        q = q_batch[batch_idx]
        lower_q, upper_q = hand.pk_chain.get_joint_limits()
        outer_q, inner_q = q.clone(), q.clone()
        for joint_name, dot in joint_dots.items():
            idx = joint_orders.index(joint_name)
            if robot_name == 'robotiq_3finger':  # open -> upper, close -> lower
                outer_q[idx] += 0.25 * ((outer_q[idx] - lower_q[idx]) if dot <= 0 else (outer_q[idx] - upper_q[idx]))
                inner_q[idx] += 0.15 * ((inner_q[idx] - upper_q[idx]) if dot <= 0 else (inner_q[idx] - lower_q[idx]))
            else:  # open -> lower, close -> upper
                outer_q[idx] += 0.25 * ((lower_q[idx] - outer_q[idx]) if dot >= 0 else (upper_q[idx] - outer_q[idx]))
                inner_q[idx] += 0.15 * ((upper_q[idx] - inner_q[idx]) if dot >= 0 else (lower_q[idx] - inner_q[idx]))
        outer_q_batch.append(outer_q)
        inner_q_batch.append(inner_q)

    outer_q_batch = torch.stack(outer_q_batch, dim=0)
    inner_q_batch = torch.stack(inner_q_batch, dim=0)

    if q_para.ndim == 2:  # batch
        return outer_q_batch.to(q_para.device), inner_q_batch.to(q_para.device)
    else:
        return outer_q_batch[0].to(q_para.device), inner_q_batch[0].to(q_para.device)
