"""
预测-验证 实验脚本
================================================================
用法：
  1. 先读每道题的说明
  2. 把你的预测填进下面的 PREDICTIONS 字典（留 None 会提示你还没填）
  3. 再运行脚本，程序自动对比你的预测和真实答案

为什么要这样学：
  "跟着敲代码"时大脑不参与，因为输入是别人给的。
  "先预测再验证"时大脑被迫先建一个模型。
  猜错最有价值 —— 猜错的那一刻，才是理解真正发生的地方。
================================================================
"""

import mujoco
import numpy as np
from pathlib import Path

# 相对脚本自身定位，这样在哪个目录下运行都能找到模型
MODEL_PATH = str(Path(__file__).resolve().parent.parent
                 / "assets/mujoco_menagerie/franka_emika_panda/scene.xml")

# ============ 第一步：把你的预测填在这里 ============
PREDICTIONS = {
    # 实验1：qpos 数组一共多长？（填一个整数）
    "exp1_qpos_len": None,

    # 实验2：只把 joint1 转 0.5 rad，末端 hand 的 z 高度会变吗？（填 True / False）
    "exp2_z_changes": None,

    # 实验3：只把 joint1 转 0.5 rad，哪些 body 的位置会变？填列表，从这些里挑：
    #        ["link1","link2","link3","link4","link5","link6","link7","hand","left_finger","right_finger"]
    "exp3_moving_bodies": None,

    # 实验4：单独把某一个关节转 0.3 rad，哪个关节让末端 hand 移动得最远？（填 1~7）
    "exp4_strongest_joint": None,

    # 实验5：连续调用 100 次 mj_forward，qpos 会变吗？（填 True / False）
    "exp5_qpos_changes_after_forward": None,
}
# ===================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
BODY_NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


def set_arm(angles):
    """设置前 7 个手臂关节角，然后只做前向计算（不推进时间）。"""
    data.qpos[:] = 0.0
    data.qpos[:7] = angles
    mujoco.mj_forward(model, data)


def joint_world_axis(j):
    """关节 j 的转轴在世界坐标系下的方向向量"""
    body = model.jnt_bodyid[j]
    return data.xmat[body].reshape(3, 3) @ model.jnt_axis[j]


def joint_world_anchor(j):
    """关节 j 的转轴经过的世界坐标点（此模型里 jnt_pos 全为 0，就是 body 原点）"""
    body = model.jnt_bodyid[j]
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ model.jnt_pos[j]


def dist_to_axis(point, anchor, axis):
    """点到一条（过 anchor、方向为 axis 的）直线的垂直距离"""
    u = axis / np.linalg.norm(axis)
    v = point - anchor
    return float(np.linalg.norm(v - np.dot(v, u) * u))


def report(no, question, guess, answer, explain, extra=""):
    print(f"\n{'=' * 66}\n实验 {no}：{question}\n{'-' * 66}")
    if extra:
        print(extra)
    print(f"  你的预测 : {guess}")
    print(f"  真实答案 : {answer}")
    if guess is None:
        print("  >>> 你还没填预测，先去上面 PREDICTIONS 里填上")
    elif guess == answer:
        print("  >>> 正确 ✅")
    else:
        print("  >>> 不一致 ❌ —— 这就是你的理解缺口，读下面的解释")
    print(f"  解释：{explain}")


print("\n" + "=" * 66)
print("  MuJoCo 心智模型 · 预测-验证实验")
print("=" * 66)

# ---------------- 实验 1 ----------------
report(1, "qpos 数组一共多长？", PREDICTIONS["exp1_qpos_len"], model.nq,
       f"model.nq = {model.nq}，等于 7 个手臂关节 + 2 个手指关节。\n"
       "        物理引擎内部把所有关节角拼成【一个一维数组】qpos，因为后面全是矩阵运算。\n"
       "        model.jnt_qposadr[k] 告诉你第 k 个关节的角度存在 qpos 的第几位。")

# ---------------- 实验 2 ----------------
set_arm([0, 0, 0, 0, 0, 0, 0])
p0 = data.xpos[HAND].copy()
set_arm([0.5, 0, 0, 0, 0, 0, 0])
p1 = data.xpos[HAND].copy()
report(2, "只转 joint1，末端 hand 的 z 高度会变吗？", PREDICTIONS["exp2_z_changes"],
       bool(not np.allclose(p0[2], p1[2])),
       f"joint1 的转轴是 {model.jnt_axis[0]}，也就是竖直的 z 轴。\n"
       f"        绕竖直轴旋转只改变水平位置，高度不变（z 一直是 {p0[2]:.4f}）。\n"
       f"        x/y 变了：({p0[0]:.4f}, {p0[1]:.4f}) -> ({p1[0]:.4f}, {p1[1]:.4f})\n"
       f"        验算：0.088×cos(0.5)={0.088 * np.cos(0.5):.4f}，"
       f"0.088×sin(0.5)={0.088 * np.sin(0.5):.4f} —— 完全吻合。\n"
       "        >>> 这就是'关节角 -> 末端位置'的映射，它的名字叫【正运动学 FK】。")

# ---------------- 实验 3 ----------------
set_arm([0, 0, 0, 0, 0, 0, 0])
before = data.xpos.copy()
tbl = "\n".join(
    f"    {BODY_NAMES[i]:<14}原点=({data.xpos[i][0]:>7.4f},{data.xpos[i][1]:>7.4f},"
    f"{data.xpos[i][2]:>6.3f})   到 z 轴的距离={np.hypot(data.xpos[i][0], data.xpos[i][1]):>7.4f}"
    for i in range(1, model.nbody))
set_arm([0.5, 0, 0, 0, 0, 0, 0])
after = data.xpos.copy()
moved = [BODY_NAMES[i] for i in range(model.nbody) if not np.allclose(before[i], after[i])]
report(3, "只转 joint1，哪些 body 的位置变了？", PREDICTIONS["exp3_moving_bodies"], moved,
       "看下面那张表：'到 z 轴的距离'为 0 的 body，转 joint1 时原点原地不动；\n"
       "        距离不为 0 的才会绕圈扫过去。\n"
       "        关键：关节转动时，它下游【整棵子树都在转】，\n"
       "        但只有【不在转轴上的点】才会产生位移 —— 转轴上的点是不动的。",
       extra=f"  qpos=0 时各 body 在世界系的位置：\n{tbl}")

# ---------------- 实验 4 ----------------
set_arm([0, 0, 0, 0, 0, 0, 0])
base = data.xpos[HAND].copy()
dists, radii = {}, {}
for j in range(7):
    ang = [0.0] * 7
    ang[j] = 0.3
    set_arm(ang)
    dists[j + 1] = float(np.linalg.norm(data.xpos[HAND] - base))

set_arm([0, 0, 0, 0, 0, 0, 0])
for j in range(7):
    radii[j + 1] = dist_to_axis(data.xpos[HAND], joint_world_anchor(j), joint_world_axis(j))

rows = "\n".join(
    f"    joint{j}  到转轴距离 r={radii[j]:.4f} m   "
    f"预测位移 r×0.3={radii[j] * 0.3:.5f}   "
    f"实测位移={dists[j]:.5f}   "
    f"轴方向={np.round(np.round([abs(v) for v in joint_world_axis(j - 1)], 0).astype(int), 0)}"
    for j in range(1, 8))
strongest = max(dists, key=dists.get)
report(4, "哪个关节让末端移动得最远？", PREDICTIONS["exp4_strongest_joint"], strongest,
       "规律非常简单：\n"
       "        位移 ≈ (末端到该关节转轴的垂直距离 r) × (转角)\n"
       "        这个 r 就是【雅可比矩阵】的本质 —— 力臂。r 越大，转一点就移动很多。\n"
       "        注意 joint7 的位移是 0：末端正好落在它的转轴上（r=0），\n"
       "        转 joint7 只是让夹爪自转，末端点原地不动。这就是「奇异位形」的雏形。",
       extra=f"  单独转每个关节 0.3 rad，末端 hand 的位移：\n{rows}")

# ---------------- 实验 5 ----------------
set_arm([0, 0, 0, 0, 0, 0, 0])
q0 = data.qpos.copy()
for _ in range(100):
    mujoco.mj_forward(model, data)
q_after_fwd = data.qpos.copy()
changed_fwd = not np.allclose(q0, q_after_fwd)

set_arm([0, 0, 0, 0, 0, 0, 0])
for _ in range(100):
    mujoco.mj_step(model, data)
q_after_step = data.qpos.copy()
drift = np.abs(q_after_step[:7] - np.zeros(7)).max()

report(5, "连续调用 100 次 mj_forward，qpos 会变吗？", PREDICTIONS["exp5_qpos_changes_after_forward"],
       changed_fwd,
       "不会变 —— mj_forward 是【纯计算】，它只根据当前 qpos 算出所有位姿和力，\n"
       "        不修改 qpos，也不推进时间。它是无副作用的。\n"
       "        而 mj_step 会真的把仿真往前推一个时间步：\n"
       f"        同样 100 次 mj_step 后，关节角最大漂移了 {drift:.6f} rad（重力让手臂下沉了一点）。\n"
       "        >>> 记住这个区分，它会决定你后面调试时用哪个函数。")

print(f"\n{'=' * 66}")
print(" 全部实验结束。猜错的那些，就是你接下来要补的地方。")
print("=" * 66)
