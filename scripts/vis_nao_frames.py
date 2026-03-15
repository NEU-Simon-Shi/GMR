#!/usr/bin/env python3
"""Visualize and print all MuJoCo body points usable for retargeting."""
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "nao" / "nao_scene.xml"
AXIS_LENGTH = 0.06


def draw_frame_at_body(viewer, data, body_id, body_name, size=AXIS_LENGTH):
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mj.mjv_initGeom(geom, type=mj.mjtGeom.mjGEOM_ARROW, size=[0.008, 0.008, 0.008],
                        pos=pos, mat=mat.flatten(), rgba=rgba_list[i])
        if i == 0:
            geom.label = body_name
        mj.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW, width=0.004,
            from_=pos, to=pos + size * mat[:, i],
        )
        viewer.user_scn.ngeom += 1


def get_retargetable_bodies(model):
    """All MuJoCo bodies except 'world' can be used as frame_type='body' targets."""
    body_names = []
    for body_id in range(model.nbody):
        body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None or body_name == "world":
            continue
        body_names.append(body_name)
    return body_names


def print_candidate_positions(model, data, body_names):
    print("=== Retargetable NAO body points (world coordinates) ===")
    for body_name in body_names:
        body_id = model.body(body_name).id
        parent_id = model.body_parentid[body_id]
        parent_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, parent_id)
        pos = data.xpos[body_id]
        print(
            f"{body_name:12s} "
            f"id={body_id:2d} "
            f"parent={parent_name:12s} "
            f"xyz=[{pos[0]: .4f}, {pos[1]: .4f}, {pos[2]: .4f}]"
        )


def main():
    model = mj.MjModel.from_xml_path(str(MODEL_PATH))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    body_names = get_retargetable_bodies(model)
    print_candidate_positions(model, data, body_names)

    viewer = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1  # hide robot, only axes
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = data.xpos[model.body("torso").id]
    viewer.cam.elevation = -15

    while viewer.is_running():
        mj.mj_forward(model, data)
        viewer.user_scn.ngeom = 0
        for body_name in body_names:
            draw_frame_at_body(viewer, data, model.body(body_name).id, body_name)
        viewer.sync()

    viewer.close()


if __name__ == "__main__":
    main()
