"""
Generate JSON scene files for pySPlisHSPlasH simulations
5 scenarios with dynamic rigid bodies and fluid coupling
"""

import json
import os
from pathlib import Path

# Base configuration
BASE_CONFIG = {
    "Configuration": {
        "pause": False,
        "sim2D": False,  # 3D simulation
        "particleRadius": 0.025,
        "numberOfStepsPerRenderUpdate": 1,
        "density0": 1000,
        "simulationMethod": 4,  # DFSPH (best for incompressibility)
        "gravitation": [0, -9.81, 0],
        "cflMethod": 1,
        "cflFactor": 1,
        "cflMaxTimeStepSize": 0.005,
        "maxIterations": 100,
        "maxError": 0.05,
        "maxIterationsV": 100,
        "maxErrorV": 0.1,
        "enableDivergenceSolver": True,
        "particleAttributes": "velocity;density",
        "renderWalls": 1,
        "boundaryHandlingMethod": 2,  # Akinci2012
        "dataExportFPS": 100,  # Export 100 frames per second
        "enablePartioExport": True,
        "partioAttributes": "position;velocity;density"
    },
    "Materials": [
        {
            "id": "Water",
            "density0": 1000,
            "colorField": "velocity",
            "colorMapType": 1,
            "renderMaxValue": 5.0,
            "surfaceTension": 0.05,
            "surfaceTensionMethod": 0,
            "viscosity": 0.01,
            "viscosityMethod": 3  # XSPH
        }
    ]
}

def create_falling_ball_scene(output_path, scenario_idx):
    """Falling ball into water"""
    scene = json.loads(json.dumps(BASE_CONFIG))

    scene["Configuration"]["stopAt"] = 0.8  # 80 frames at 100 FPS

    # Container walls (static rigid body)
    scene["RigidBodies"] = [
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.5, 0.5],
            "scale": [1.0, 1.0, 1.0],
            "color": [0.1, 0.1, 0.1, 0.3],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapThickness": 0.0,
            "mapResolution": [25, 25, 25]
        },
        # Dynamic sphere
        {
            "geometryFile": "../models/UnitSphere.obj",
            "translation": [0.5, 0.85, 0.5],
            "scale": [0.15, 0.15, 0.15],
            "color": [0.8, 0.2, 0.2, 1.0],
            "isDynamic": True,
            "isWall": False,
            "density": 1500,  # Heavier than water
            "mapResolution": [15, 15, 15]
        }
    ]

    # Water block
    scene["FluidBlocks"] = [
        {
            "id": "Water",
            "start": [0.1, 0.05, 0.1],
            "end": [0.9, 0.5, 0.9]
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"Created: {output_path}")

def create_floating_ball_scene(output_path, scenario_idx):
    """Light ball floating up"""
    scene = json.loads(json.dumps(BASE_CONFIG))

    scene["Configuration"]["stopAt"] = 0.8

    scene["RigidBodies"] = [
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.5, 0.5],
            "scale": [1.0, 1.0, 1.0],
            "color": [0.1, 0.1, 0.1, 0.3],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapResolution": [25, 25, 25]
        },
        # Light sphere (will float)
        {
            "geometryFile": "../models/UnitSphere.obj",
            "translation": [0.5, 0.2, 0.5],
            "scale": [0.12, 0.12, 0.12],
            "color": [0.2, 0.8, 0.2, 1.0],
            "isDynamic": True,
            "isWall": False,
            "density": 500,  # Lighter than water
            "mapResolution": [15, 15, 15]
        }
    ]

    scene["FluidBlocks"] = [
        {
            "id": "Water",
            "start": [0.1, 0.05, 0.1],
            "end": [0.9, 0.6, 0.9]
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"Created: {output_path}")

def create_heavy_box_scene(output_path, scenario_idx):
    """Heavy box sinking"""
    scene = json.loads(json.dumps(BASE_CONFIG))

    scene["Configuration"]["stopAt"] = 0.8

    scene["RigidBodies"] = [
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.5, 0.5],
            "scale": [1.0, 1.0, 1.0],
            "color": [0.1, 0.1, 0.1, 0.3],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapResolution": [25, 25, 25]
        },
        # Heavy box
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.85, 0.5],
            "scale": [0.15, 0.15, 0.15],
            "color": [0.5, 0.5, 0.8, 1.0],
            "isDynamic": True,
            "isWall": False,
            "density": 2000,
            "mapResolution": [15, 15, 15]
        }
    ]

    scene["FluidBlocks"] = [
        {
            "id": "Water",
            "start": [0.1, 0.05, 0.1],
            "end": [0.9, 0.5, 0.9]
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"Created: {output_path}")

def create_multi_ball_scene(output_path, scenario_idx):
    """Multiple balls with different densities"""
    scene = json.loads(json.dumps(BASE_CONFIG))

    scene["Configuration"]["stopAt"] = 0.8

    scene["RigidBodies"] = [
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.5, 0.5],
            "scale": [1.0, 1.0, 1.0],
            "color": [0.1, 0.1, 0.1, 0.3],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapResolution": [25, 25, 25]
        },
        # Ball 1 - Heavy
        {
            "geometryFile": "../models/UnitSphere.obj",
            "translation": [0.35, 0.8, 0.5],
            "scale": [0.12, 0.12, 0.12],
            "color": [0.8, 0.2, 0.2, 1.0],
            "isDynamic": True,
            "density": 1500,
            "mapResolution": [12, 12, 12]
        },
        # Ball 2 - Light
        {
            "geometryFile": "../models/UnitSphere.obj",
            "translation": [0.65, 0.8, 0.5],
            "scale": [0.1, 0.1, 0.1],
            "color": [0.2, 0.8, 0.2, 1.0],
            "isDynamic": True,
            "density": 600,
            "mapResolution": [10, 10, 10]
        }
    ]

    scene["FluidBlocks"] = [
        {
            "id": "Water",
            "start": [0.1, 0.05, 0.1],
            "end": [0.9, 0.45, 0.9]
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"Created: {output_path}")

def create_bouncing_sphere_scene(output_path, scenario_idx):
    """Sphere bouncing in water"""
    scene = json.loads(json.dumps(BASE_CONFIG))

    scene["Configuration"]["stopAt"] = 0.8

    scene["RigidBodies"] = [
        {
            "geometryFile": "../models/UnitBox.obj",
            "translation": [0.5, 0.5, 0.5],
            "scale": [1.0, 1.0, 1.0],
            "color": [0.1, 0.1, 0.1, 0.3],
            "isDynamic": False,
            "isWall": True,
            "mapInvert": True,
            "mapResolution": [25, 25, 25]
        },
        # Bouncing sphere
        {
            "geometryFile": "../models/UnitSphere.obj",
            "translation": [0.5, 0.85, 0.5],
            "scale": [0.15, 0.15, 0.15],
            "color": [0.8, 0.5, 0.2, 1.0],
            "isDynamic": True,
            "density": 1200,
            "restitution": 0.5,  # Some bounce
            "mapResolution": [15, 15, 15]
        }
    ]

    scene["FluidBlocks"] = [
        {
            "id": "Water",
            "start": [0.15, 0.05, 0.15],
            "end": [0.85, 0.4, 0.85]
        }
    ]

    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)

    print(f"Created: {output_path}")

def main():
    print("=" * 70)
    print("GENERATING PYSPLISHSPLASH SCENE FILES")
    print("5 scenarios with dynamic rigid bodies")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("datasets/sph_scenes")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("falling_ball", create_falling_ball_scene),
        ("floating_ball", create_floating_ball_scene),
        ("heavy_box", create_heavy_box_scene),
        ("multi_ball", create_multi_ball_scene),
        ("bouncing_sphere", create_bouncing_sphere_scene)
    ]

    total_scenes = 0

    for scenario_name, create_func in scenarios:
        print(f"\n{scenario_name.upper()}:")
        for idx in range(5):
            scene_file = output_dir / f"sph_{scenario_name}_{idx:03d}.json"
            create_func(scene_file, idx)
            total_scenes += 1

    print()
    print("=" * 70)
    print(f"SCENE GENERATION COMPLETE: {total_scenes} scenes created")
    print(f"Output: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
