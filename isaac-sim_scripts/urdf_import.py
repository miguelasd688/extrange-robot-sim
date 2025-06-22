from isaacsim.simulation_app import SimulationApp

# Inicializar simulación
kit = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
    "use_full_gpu": True,
    "enable_ui": False
})

import time
from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleArticulation
from omni.usd import get_context
from pxr import Usd

# Ruta al archivo USD
usd_path = "/home/miguel/IsaacSim/strange-robot/1-robot-env.usd"


# Cargar el escenario
get_context().open_stage(usd_path)
world = World(physics_dt=1/60)
stage = world.stage

if not stage:
    raise RuntimeError(".usd file could not be loaded")

print("Loading .usd file")

# Esperar que el prim del robot esté disponible
robot_root_path = "/World/extrange_robot"
timeout = 10.0
start_time = time.time()
robot_prim = stage.GetPrimAtPath(robot_root_path)

while (not robot_prim.IsValid() or not Usd.Prim.IsDefined(robot_prim)) and time.time() - start_time < timeout:
    print("Waiting to load robot...")
    kit.update()
    robot_prim = stage.GetPrimAtPath(robot_root_path)

if not robot_prim.IsValid():
    raise RuntimeError(f"No robot found at {robot_root_path}")
print(f"Robot found at {robot_root_path}")

# Inicializar simulación
world.reset()
kit.update()

# Esperar que física esté activa
timeout = 5.0
start_time = time.time()
while not world.is_playing() and (time.time() - start_time < timeout):
    print("Waiting for physics...")
    kit.update()

# Ruta al articulation root real
robot_articulation_path = "/World/extrange_robot/base_link"
robot = SingleArticulation(prim_path=robot_articulation_path)
robot.initialize()

if not robot.is_valid():
    raise RuntimeError("Robot has no valid articulation")
print(f"Articulation initialized at {robot_articulation_path}")
print("Abailable joints:", robot.dof_names)

# Suponemos que quieres girar la primera articulación
joint_index = 0
target_velocity = 10.0  # rad/s

# Activar modo de control de velocidad
robot.set_joint_positions([0.0] * robot.num_dof)  # Inicializa posiciones (por seguridad)
robot.set_joint_velocities([0.0] * robot.num_dof)         # Asegura estado limpio
robot.set_joint_efforts([10.0] * robot.num_dof)            # Limpia torques


# Bucle de simulación
for _ in range(1000):
    # Aplicar velocidad en esa articulación
    velocities = [0.0] * robot.num_dof
    velocities[joint_index] = target_velocity
    robot.set_joint_velocities(velocities)

    world.step(render=True)

# Cierre limpio
kit.close()
