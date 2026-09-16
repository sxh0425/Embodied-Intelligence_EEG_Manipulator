"""
热身关卡 · 从零搭台阶到 EX5
================================================================
为什么有这个文件：
  exercises.py 从 EX4 开始，一道题就要同时用 5 个你还没见过的 API。
  那不是"难"，是"跳步"。这个文件把台阶拆开，一级只加一个新东西。

用法：
  一次只做一级。做完一级立刻运行看结果，通过了再做下一级。
  每级都给了确切的字段名 —— 你不需要猜 API，只需要想清楚逻辑。

运行：python robotics/warmup.py
================================================================
"""

import mujoco
import numpy as np
from pathlib import Path

MODEL_PATH = str(Path(__file__).resolve().parent.parent
                 / "assets/mujoco_menagerie/franka_emika_panda/scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ==================== 第 0 级：示范（已经写好，读一遍就行）====================
def level0():
    """目标：见识"探索式编程"长什么样。

    不知道一个对象有什么字段时，用 dir() 打印出来看。
    这不是作弊，这是日常工作方式。
    """
    names = [n for n in dir(model) if not n.startswith("_")]
    print(f"    model 有 {len(names)} 个公开字段，前 20 个：")
    print(f"    {names[:20]}")
    print("    （你可以看到 nq / nv / nu / njnt / nbody / jnt_axis / jnt_pos ...）")
    return len(names)


# ==================== 第 1 级：读 model 的三个尺寸 ====================
def level1():
    """目标：返回一个元组 (nq, nv, nu)。

    提示：字段名就叫 model.nq / model.nv / model.nu，直接读。
    """
    ######## TODO ########
    nq = model.nq
    nv = model.nv
    nu = model.nu
    return (nq,nv,nu)


# ==================== 第 2 级：qpos 有多长 ====================
def level2():
    """目标：返回 data.qpos 的长度。

    提示：data.qpos 是个 numpy 数组，用 len() 就能量长度。
          不要写死 9，用代码算出来。
    """
    ######## TODO ########
    len_qpos = len(data.qpos)
    return len_qpos


# ==================== 第 3 级：让引擎算一次 ====================
def level3():
    """目标：先调用 mj_forward，然后返回 data.xpos 的形状（shape）。

    提示：mj_forward(model, data) —— 只算，不推进时间。
          返回值是 array.shape，会是一个元组，类似 (12, 3)。
          第一个数是 body 个数，第二个数是 3（x, y, z）。
    """
    ######## TODO ########
    mujoco.mj_forward(model, data)
    return data.xpos.shape


# ==================== 第 4 级：读一个 body 的位置 ====================
def level4():
    """目标：返回 'hand' 这个 body 的世界坐标（长度 3 的数组）。

    提示：先用 mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
          把名字换成编号，再用编号去索引 data.xpos。
    """
    ######## TODO ########
    id_hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    print(id_hand)
    print(type(id_hand))
    pos_hand = data.xpos[id_hand]
    return pos_hand


# ==================== 第 5 级：读朝向（reshape 陷阱）====================
def level5():
    """目标：返回 'hand' body 的朝向，形状必须是 (3, 3) 的矩阵。

    提示：data.xmat 里每个 body 的朝向被【展开成 9 个数】存着，
          所以要先 .reshape(3, 3) 才能当矩阵用。
          参考 01_load_panda.py 里你写过的那行。
    """
    ######## TODO ########
    id_hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return data.xmat[id_hand].reshape(3,3)


# ==================== 第 6 级：关节挂在哪个 body 上 ====================
def level6():
    """目标：返回关节 0（也就是 joint1）所属 body 的【名字】。

    提示：model.jnt_bodyid[0] 给出 body 的编号。
          再用 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 编号) 换成名字。
    """
    ######## TODO ########
    body_id = model.jnt_bodyid[0]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    print(body_name)
    return body_name


# ==================== 第 7 级：读关节的轴（局部坐标）====================
def level7():
    """目标：返回关节 0 的转轴方向，形状是 (3,)。

    提示：字段是 model.jnt_axis[0]，直接读。
          注意！这个轴是在【body 的局部坐标系】里，不是世界坐标系。
          下一级就来解决这个问题。
    """
    ######## TODO ########
    axis = model.jnt_axis[0]
    print(axis)
    print(type(axis))
    return axis


# ==================== 第 8 级：把轴转到世界坐标系 ★ 关键台阶 ====================
def level8():
    """目标：返回关节 0 的转轴在【世界坐标系】下的方向。

    提示：这是整套题里最关键的一步，公式是：

             世界轴 = body 的朝向矩阵  ×  局部轴

          其中：
            body 的朝向矩阵 = data.xmat[body编号].reshape(3, 3)
            body 编号       = model.jnt_bodyid[0]
            局部轴          = model.jnt_axis[0]

          另外：关节0的body是link1，它的朝向是单位矩阵，
          所以这一步的结果会和你上一级看到的一样 [0, 0, 1]。
          别被"看起来没变"骗了 —— 换个关节（比如关节1）就会不一样。

    自检：算完之后可以直接对拍 —— MuJoCo 已经把标准答案放在 data.xaxis[0] 里了。
          print(np.allclose(你的结果, data.xaxis[0], atol=1e-12))
    """
    ######## TODO ########
    axis = model.jnt_axis[0]
    body_id = model.jnt_bodyid[0]
    mat = data.xmat[body_id].reshape(3,3)
    world_pos = mat @ axis
    print(world_pos)
    return world_pos


# ==================== 第 9 级：转轴经过哪个点 ====================
def level9():
    """目标：返回关节 0 的转轴所经过的世界坐标点，形状是 (3,)。

    提示：公式：

             世界位置 = body 的世界位置  +  body 朝向矩阵 × 局部位置

          其中：
            body 的世界位置 = data.xpos[body编号]
            body 朝向矩阵   = data.xmat[body编号].reshape(3, 3)
            局部位置        = model.jnt_pos[0]

          结构是不是和第 8 级一模一样？只是把"方向"换成了"位置"。

    自检：标准和答案在 data.xanchor[0]，同样可以对拍验证。
    """
    ######## TODO ########
    body_id = model.jnt_bodyid[0]
    body_mat = data.xmat[body_id].reshape(3,3)
    body_pos = data.xpos[body_id]
    joint_pos = model.jnt_pos[0]
    joint_world_pos = body_pos + body_mat @ joint_pos
    return joint_world_pos


# ==================== 第 10 级：点到直线的距离 ====================
def level10_dist_to_line(P, A, u):
    """目标：求点 P 到【过点 A、方向为 u 的直线】的垂直距离。

    这一级是纯数学，和 MuJoCo 无关。想清楚几何再写。

    提示：直线上任意一点都能写成 A + t·u。
          先算 v = P - A（从 A 指向 P 的向量）。
          把 v 投影到 u 上，得到投影向量 proj。
          垂足 = A + proj，所以垂直距离 = ||v - proj||。

          投影向量的公式：proj = (v·u) / (u·u) * u
          如果 u 已经是单位向量，就简化为 proj = (v·u) * u

          先用 np.linalg.norm(u) 把 u 单位化，再用简化公式。
    """
    ######## TODO ########
    u = u / np.linalg.norm(u)
    v = P - A
    proj = np.dot(u,v) * u
    return np.linalg.norm(v - proj)


# ==================== 第 11 级：力臂 = EX5 的答案 ====================
def level11_lever_arm(joint_index):
    """目标：返回末端 hand 到关节 joint_index 转轴的垂直距离（力臂 r）。

    这就是 exercises.py 里 EX5 要的东西。你现在已经集齐所有零件了：

      第 9 级  → 转轴经过的点 A
      第 8 级  → 转轴的方向 u（记得换成 joint_index）
      第 4 级  → 末端点 P
      第 10 级 → 点到直线的距离

    提示：把上面四个拼起来。
          注意用 joint_index 替换掉写死的 0。
    """
    ######## TODO ########
    return None


# ==================== 第 12 级：手写欧拉积分 ====================
def level12_euler(n_steps, dt):
    """目标：手写自由落体，返回第 n_steps 步之后的 z 高度。

    设定：从 z = 10.0 开始，初速度 0，重力 g = -9.81，时间步 dt。

    提示：循环里就两个式子，而且【顺序很重要】：

            v = v + g * dt      ← 先更新速度
            z = z + v * dt      ← 再用新速度更新位置

          MuJoCo 用的就是这个顺序（叫"半隐式欧拉"）。
          如果把顺序反过来，结果就会和 MuJoCo 对不上。
    """
    z, v, g = 10.0, 0.0, -9.81
    ######## TODO ########
    return None


# ==================== 自动检查（不要改）====================

def _eq(got, want, tol=1e-9, label=""):
    if got is None:
        return None
    try:
        if isinstance(want, str):
            return str(got) == want
        if isinstance(want, (int, float)):
            return abs(float(got) - float(want)) <= tol
        return np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), atol=tol)
    except Exception:
        return False


def _run(no, title, fn, want, tol=1e-9, note=""):
    got = fn()
    if got is None:
        print(f"  第 {no:>2} 级  {title:<26} ⬜ 还没写")
        return False
    ok = _eq(got, want, tol)
    print(f"  第 {no:>2} 级  {title:<26} {'✅' if ok else '❌'}")
    if not ok:
        print(f"          你的答案 : {got}")
        print(f"          正确答案 : {want}")
    elif note:
        print(f"          {note}")
    return ok


def main():
    print("\n" + "=" * 64)
    print("  热身关卡（做完这个再去 exercises.py）")
    print("=" * 64)

    mujoco.mj_forward(model, data)
    HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    r = []
    print("\n第 0 级是示范，直接跑给你看：")
    level0()

    print()
    r.append(_run(1, "model 的三个尺寸", level1, (9, 9, 8)))
    r.append(_run(2, "qpos 的长度", level2, 9))
    r.append(_run(3, "xpos 的形状", level3, (12, 3)))
    r.append(_run(4, "hand 的世界位置", level4, data.xpos[HAND], 1e-6))
    r.append(_run(5, "hand 的朝向矩阵", level5, data.xmat[HAND].reshape(3, 3), 1e-6))
    r.append(_run(6, "关节0所属 body 的名字", level6, "link1"))
    r.append(_run(7, "关节0的局部轴", level7, model.jnt_axis[0], 1e-9))
    r.append(_run(8, "关节0的世界轴", level8,
                  data.xmat[model.jnt_bodyid[0]].reshape(3, 3) @ model.jnt_axis[0], 1e-9))
    r.append(_run(9, "关节0转轴经过的点", level9,
                  data.xpos[model.jnt_bodyid[0]]
                  + data.xmat[model.jnt_bodyid[0]].reshape(3, 3) @ model.jnt_pos[0], 1e-9))

    ok10 = level10_dist_to_line(np.array([1.0, 1.0, 0.0]),
                                np.array([0.0, 0.0, 0.0]),
                                np.array([1.0, 0.0, 0.0]))
    r.append(_run(10, "点到直线的距离", lambda: ok10, 1.0, 1e-9,
                  note="点(1,1,0) 到 x 轴的距离应该是 1，正好是它的 y 坐标。"))

    print()
    ok11 = True
    for j in range(7):
        body = model.jnt_bodyid[j]
        A = data.xpos[body] + data.xmat[body].reshape(3, 3) @ model.jnt_pos[j]
        u = data.xmat[body].reshape(3, 3) @ model.jnt_axis[j]
        u = u / np.linalg.norm(u)
        v = data.xpos[HAND] - A
        want = float(np.linalg.norm(v - np.dot(v, u) * u))
        ok11 &= _run(11, f"力臂 r (joint{j + 1})", (lambda jj=j: level11_lever_arm(jj)), want, 1e-6)
    r.append(ok11)

    print()
    m2 = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.001"/>
      <worldbody>
        <body name="ball" pos="0 0 10"><freejoint/><geom type="sphere" size="0.05"/></body>
      </worldbody>
    </mujoco>""")
    d2 = mujoco.MjData(m2)
    for _ in range(1000):
        mujoco.mj_step(m2, d2)
    r.append(_run(12, "欧拉积分 · 1000 步后的 z", lambda: level12_euler(1000, m2.opt.timestep),
                  float(d2.qpos[2]), 1e-3,
                  note="和 MuJoCo 精确吻合 —— 因为它用的就是这个积分顺序。"))

    n = sum(1 for x in r if x)
    print("\n" + "=" * 64)
    print(f"  完成度：{n} / {len(r)}")
    if n == len(r):
        print("  全过了 🎉 现在去 exercises.py，你会发现 EX1~EX7 变得简单了。")
    else:
        print("  还没全过。一次只攻一级，卡住了看那一级的提示。")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
