from isaacsim.simulation_app import SimulationApp

def LaunchSimulationApp():
    return SimulationApp({
        "headless": False,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
        "use_full_gpu": True,
        "enable_ui": False
    })