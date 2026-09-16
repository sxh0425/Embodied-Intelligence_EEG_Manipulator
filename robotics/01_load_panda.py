import mujoco
import numpy as np

MODEL_PATH = "../assets/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("nq",model.nq)
print("nv",model.nv)
print("nu",model.nu)

for i in range(model.njnt):
    name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{i}:{name} qposadr={model.jnt_qposadr[i]}")

for i in range(model.nbody):
    print(f"{i}:{mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i)}")

mujoco.mj_forward(model,data)

hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
print(f"\nhand body id = {hid}")
print("位置:", np.round(data.xpos[hid], 4))
print("姿态矩阵:\n", np.round(data.xmat[hid].reshape(3, 3), 4))
