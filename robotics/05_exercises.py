"""
练习题 · 带自动检查
================================================================
用法：
  1. 找到标着 ######## TODO ######## 的地方
  2. 用你自己的代码把 return 后面补上（别改函数名和参数）
  3. 运行 python robotics/05_exercises.py
  4. 全部通过会打印 PASS

对应课程（在课程/ 目录里）：
  课程/第04课-用名字找到编号.md      读位置
  课程/第05课-朝向为什么是9个数.md     读朝向
  课程/第08课-局部坐标和世界坐标.md    局部 → 世界
  课程/第11课-力臂.md                EX5 的完整拆解

规则：先自己写。卡住了再回去看课程/里的对应章节，不要直接问答案。
================================================================
"""

import mujoco
import numpy as np
from pathlib import Path

MODEL_PATH = str(Path(__file__).resolve().parent.parent
                 / "assets/mujoco_menagerie/franka_emika_panda/scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
HAND_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

N_ARM_JOINTS = 7


def set_arm(angles):
    """把前 7 个手臂关节设成 angles，其余关节置 0，然后做一次前向计算。

    注意用的是 mj_forward 而不是 mj_step —— 只算，不推进时间。
    """
    data.qpos[:] = 0.0
    data.qpos[:N_ARM_JOINTS] = angles
    mujoco.mj_forward(model, data)


# EX1 ------------------------------------------------------------------
def ex1_state_vector_size(data):
    """返回 qpos 数组的长度。

    提示：不要写死数字，从 model 里读出来。
    """
    ######## TODO ########
    len_qpos = len(data.qpos)
    return len_qpos


# EX2 ------------------------------------------------------------------
def ex2_no_arm_joints():
    """返回：这个模型里手臂关节一共有几个。

    提示：手臂关节是 joint1..joint7，手指关节也算关节，
          但你要返回的是"手臂"的部分。
          可以从 model.jnt_type 入手：转动关节和滑动关节的类型编号不一样。
    """
    ######## TODO ########
    for i in range (9):
        if model.jnt_type[i] == joint:
            flag += 1

    return flag


# EX3 ------------------------------------------------------------------
def ex3_set_and_get_hand(angles):
    """把手臂设成 angles 这组关节角，返回末端 hand 的世界坐标（长度 3 的 numpy 数组）。

    提示：先调用上面的 set_arm，再读某个字段。
    """
    ######## TODO ########
    return None


# EX4 ------------------------------------------------------------------
def ex4_joint_anchor(joint_index):
    """返回关节 joint_index（0~6）的转轴所经过的世界坐标点。

    提示：关节挂在某个 body 上，转轴在该 body 的局部坐标系里。
          MuJoCo 里关节轴的世界位置 = body 世界位置 + body 姿态 × jnt_pos
          相关字段：model.jnt_bodyid[j]、model.jnt_pos[j]、data.xpos、data.xmat
          注意 xmat 是展开成 9 个数的 3×3 矩阵，要 reshape(3, 3) 才能做矩阵乘法。
    """
    ######## TODO ########
    return None


# EX5 ------------------------------------------------------------------
def ex5_lever_arm(joint_index):
    """返回：末端 hand 到关节 joint_index 转轴的【垂直距离】（也就是力臂 r）。

    提示：点到直线的距离。
          直线由一点 A 和方向 u 确定：直线上任意点可写成 A + t·u
          点 P 到它的垂直距离 = ||(P-A) - ((P-A)·u)·u||
          方向 u 是关节轴在世界坐标系下的方向 = body 姿态 × jnt_axis
          记得先做一次 set_arm([0]*7) 让状态干净。
    """
    ######## TODO ########
    return None


# EX6 ------------------------------------------------------------------
def ex6_predict_displacement(joint_index, theta):
    """用「力臂 × 转角」预测：单独把 joint_index 转 theta 弧度后，末端移动多远。

    提示：直接复用 ex5_lever_arm 的结果。
          近似公式：位移 ≈ 力臂 × 转角（课程/第11课讲过力臂）。
    """
    ######## TODO ########
    return None


# EX7 ------------------------------------------------------------------
def ex7_free_fall_euler(n_steps, dt):
    """手写欧拉积分，模拟自由落体，返回第 n_steps 步之后的 z 高度。

    这是本套题里唯一一道"脱离 MuJoCo"的题。
    课程/第12课讲过：仿真就是"一小步一小步地算"。

    设定：小球从 z = 10.0 开始，初速度 0，重力 g = -9.81，时间步 dt。
          一共走 n_steps 步。忽略空气阻力，不撞地。

    提示：循环里两个式子——
          速度先更新，位置后更新（这个顺序很关键，MuJoCo 用的就是这个顺序）
    """
    z = 10.0
    v = 0.0
    g = -9.81
    ######## TODO ########
    return None


# ==================== 下面是自动检查，不要改 ====================

class _Err:
    """把这个函数报的错装起来，交给 _check 统一显示，不让它中断整个程序。"""
    def __init__(self, e):
        self.e = e


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return _Err(e)


def _check(label, got, want, tol=1e-9):
    if isinstance(got, _Err):
        print(f"  {label:<34} ❌ 你的函数报错了：{type(got.e).__name__}: {got.e}")
        return False
    if got is None:
        print(f"  {label:<34} ⬜ 还没写")
        return False
    try:
        if isinstance(want, (int, float)):
            ok = abs(float(got) - float(want)) <= tol
        else:
            ok = np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), atol=tol)
    except Exception as e:
        print(f"  {label:<34} ❌ 报错：{e}")
        return False
    print(f"  {label:<34} {'✅' if ok else '❌'}")
    if not ok:
        print(f"      你的答案 : {got}")
        print(f"      正确答案 : {want}")
    return ok


def main():
    print("\n" + "=" * 58)
    print("  练习题自动检查")
    print("=" * 58)
    results = []

    print("\n[EX1] qpos 的长度")
    results.append(_check("len(qpos)", _safe(ex1_state_vector_size, data), model.nq))

    print("\n[EX2] 手臂关节数")
    results.append(_check("手臂关节数", _safe(ex2_no_arm_joints), N_ARM_JOINTS))

    print("\n[EX3] 设置关节角并读取末端位置")
    test_angles = [0.3, -0.5, 0.2, -1.0, 0.4, 1.2, 0.6]
    set_arm(test_angles)
    want_pos = data.xpos[HAND_ID].copy()
    got_pos = _safe(ex3_set_and_get_hand, test_angles)
    results.append(_check("hand 世界坐标", got_pos, want_pos, tol=1e-6))

    print("\n[EX4] 关节转轴上的世界坐标点")
    set_arm([0.0] * 7)
    ok4 = True
    for j in range(N_ARM_JOINTS):
        body = model.jnt_bodyid[j]
        want = data.xpos[body] + data.xmat[body].reshape(3, 3) @ model.jnt_pos[j]
        ok4 &= _check(f"joint{j + 1} 转轴 anchor", _safe(ex4_joint_anchor, j), want, tol=1e-6)
    results.append(ok4)

    print("\n[EX5] 力臂：末端到各关节转轴的垂直距离")
    set_arm([0.0] * 7)
    p_hand = data.xpos[HAND_ID].copy()
    ok5 = True
    for j in range(N_ARM_JOINTS):
        body = model.jnt_bodyid[j]
        A = data.xpos[body] + data.xmat[body].reshape(3, 3) @ model.jnt_pos[j]
        u = data.xmat[body].reshape(3, 3) @ model.jnt_axis[j]
        u = u / np.linalg.norm(u)
        v = p_hand - A
        want = float(np.linalg.norm(v - np.dot(v, u) * u))
        ok5 &= _check(f"joint{j + 1} 力臂 r", _safe(ex5_lever_arm, j), want, tol=1e-6)
    results.append(ok5)

    print("\n[EX6] 用「力臂 × 转角」预测位移，并和 MuJoCo 实测对比（误差 < 1%）")
    theta = 0.3
    ok6 = True
    for j in range(N_ARM_JOINTS):
        set_arm([0.0] * 7)
        base = data.xpos[HAND_ID].copy()
        angles = [0.0] * 7
        angles[j] = theta
        set_arm(angles)
        actual = float(np.linalg.norm(data.xpos[HAND_ID] - base))
        pred = _safe(ex6_predict_displacement, j, theta)
        if isinstance(pred, _Err):
            print(f"  joint{j + 1}: ❌ 你的函数报错了：{type(pred.e).__name__}: {pred.e}")
            ok6 = False
            continue
        if pred is None:
            print(f"  joint{j + 1}: ⬜ 还没写")
            ok6 = False
            continue
        rel = abs(float(pred) - actual) / max(actual, 1e-9) if actual > 1e-9 else 0.0
        good = rel < 0.01
        ok6 &= good
        print(f"  joint{j + 1}: 预测={float(pred):.5f}  实测={actual:.5f}  "
              f"相对误差={rel * 100:.2f}%  {'✅' if good else '❌'}")
    results.append(ok6)

    print("\n[EX7] 手写欧拉积分 vs MuJoCo（自由落体）")
    XML = """
    <mujoco>
      <option gravity="0 0 -9.81" timestep="0.001"/>
      <worldbody>
        <body name="ball" pos="0 0 10">
          <freejoint/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m2 = mujoco.MjModel.from_xml_string(XML)
    d2 = mujoco.MjData(m2)
    n_steps, dt = 1000, m2.opt.timestep
    for _ in range(n_steps):
        mujoco.mj_step(m2, d2)
    want_z = float(d2.qpos[2])
    got_z = _safe(ex7_free_fall_euler, n_steps, dt)
    ok7 = _check(f"{n_steps} 步后的 z 高度", got_z, want_z, tol=1e-3)
    if ok7:
        print(f"      MuJoCo 用的也是这个积分顺序（先更新速度，再更新位置），")
        print(f"      所以你的手写结果能和它精确对上。这叫【半隐式欧拉】。")
    results.append(ok7)

    n_ok = sum(1 for r in results if r)
    print("\n" + "=" * 58)
    print(f"  完成度：{n_ok} / {len(results)}")
    print("  PASS 🎉" if n_ok == len(results) else "  还没全过，继续加油。")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
