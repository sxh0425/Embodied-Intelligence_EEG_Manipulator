import mujoco
import mujoco.viewer   # 必须显式 import，只 import mujoco 是不够的

model = mujoco.MjModel.from_xml_path("../assets/mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# launch 会同时启动物理循环和可视化窗口，阻塞直到你关掉窗口
mujoco.viewer.launch(model, data)
