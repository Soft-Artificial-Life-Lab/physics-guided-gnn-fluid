"""
Generate Optimized SPH Training Data - 300 High-Quality Scenarios

Focuses on quality and diversity over quantity:
- 50 dam break variations
- 50 splash scenarios
- 50 wave simulations
- 50 pour scenarios
- 50 fountain scenarios
- 50 multiple droplet scenarios

Total: 300 scenarios, ~15,000 frames
"""
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import logging
from datetime import datetime
from tqdm import tqdm
import json

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"optimized_sph_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*60)
print("Optimized SPH Data Generation - 300 Quality Scenarios")
print("="*60)
logger.info("="*60)
logger.info("Optimized SPH Data Generation Started")
logger.info(f"Log file: {log_file}")
logger.info("="*60)


class FastSPHSimulator:
    """Optimized SPH simulator with vectorized operations"""

    def __init__(self, particle_mass=0.02, smoothing_radius=0.3,
                 gas_constant=1.0, viscosity=0.01, gravity_scale=1.0):
        self.mass = particle_mass
        self.h = smoothing_radius
        self.k = gas_constant
        self.mu = viscosity
        self.gravity = np.array([0, 0, -9.81 * gravity_scale])
        self.dt = 0.01
        self.rest_density = 1000.0

    def poly6_kernel(self, r, h):
        mask = r <= h
        coef = 315.0 / (64.0 * np.pi * h**9)
        result = np.zeros_like(r)
        result[mask] = coef * (h**2 - r[mask]**2)**3
        return result

    def compute_density_vectorized(self, positions):
        """Vectorized density computation"""
        N = len(positions)
        tree = cKDTree(positions)
        densities = np.zeros(N)

        for i in range(N):
            neighbors_idx = tree.query_ball_point(positions[i], self.h)
            if len(neighbors_idx) > 0:
                r = np.linalg.norm(positions[neighbors_idx] - positions[i], axis=1)
                densities[i] = self.mass * np.sum(self.poly6_kernel(r, self.h))

        densities = np.maximum(densities, self.rest_density)
        return densities

    def compute_forces_fast(self, positions, velocities, densities):
        """Faster force computation"""
        N = len(positions)
        forces = np.zeros((N, 3))
        tree = cKDTree(positions)

        for i in range(N):
            neighbors = tree.query_ball_point(positions[i], self.h)
            if len(neighbors) <= 1:
                forces[i] = self.mass * self.gravity
                continue

            for j in neighbors:
                if i == j:
                    continue

                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)

                if r < 1e-6:
                    continue

                # Pressure
                pressure_i = self.k * (densities[i] - self.rest_density)
                pressure_j = self.k * (densities[j] - self.rest_density)

                if r <= self.h:
                    coef = -45.0 / (np.pi * self.h**6)
                    spiky_grad = coef * (self.h - r)**2 * r_vec / r
                    forces[i] -= self.mass * (pressure_i + pressure_j) / (2.0 * densities[j]) * spiky_grad

                # Viscosity
                if r <= self.h:
                    visc_coef = 45.0 / (np.pi * self.h**6) * (self.h - r)
                    forces[i] += self.mu * self.mass * (velocities[j] - velocities[i]) / densities[j] * visc_coef

            forces[i] += self.mass * self.gravity

        return forces

    def step(self, positions, velocities):
        """Simulation step with boundary handling"""
        densities = self.compute_density_vectorized(positions)
        forces = self.compute_forces_fast(positions, velocities, densities)

        accelerations = forces / self.mass
        velocities += accelerations * self.dt
        positions += velocities * self.dt

        # Ground collision
        ground = positions[:, 2] < 0
        positions[ground, 2] = 0
        velocities[ground, 2] *= -0.5

        # Wall collisions
        for axis in [0, 1]:
            lower = positions[:, axis] < -3.0
            upper = positions[:, axis] > 3.0
            positions[lower, axis] = -3.0
            positions[upper, axis] = 3.0
            velocities[lower, axis] *= -0.5
            velocities[upper, axis] *= -0.5

        return positions, velocities


class ScenarioGenerator:
    """Generate diverse fluid scenarios"""

    @staticmethod
    def dam_break(num_particles, dam_height, dam_width, dam_position_x):
        positions = []
        n = int(np.cbrt(num_particles))

        for x in np.linspace(dam_position_x - dam_width/2, dam_position_x + dam_width/2, n):
            for y in np.linspace(-0.5, 0.5, n):
                for z in np.linspace(0.1, dam_height, n):
                    if len(positions) < num_particles:
                        positions.append([x, y, z])

        positions = np.array(positions[:num_particles], dtype=np.float32)
        positions += np.random.randn(*positions.shape) * 0.02
        velocities = np.zeros_like(positions)
        velocities[:, 0] = np.random.uniform(-0.2, 0.5, len(positions))
        return positions, velocities

    @staticmethod
    def splash(num_particles, drop_height, pool_particles=0):
        positions = []
        drop_count = num_particles - pool_particles

        # Sphere drop
        while len(positions) < drop_count:
            point = np.random.uniform(-0.3, 0.3, 3)
            if np.linalg.norm(point) <= 0.3:
                positions.append(point)

        # Pool
        if pool_particles > 0:
            n = int(np.sqrt(pool_particles))
            for x in np.linspace(-2.0, 2.0, n):
                for y in np.linspace(-2.0, 2.0, n):
                    if len(positions) < num_particles:
                        positions.append([x, y, 0.1])

        positions = np.array(positions[:num_particles], dtype=np.float32)
        positions[:drop_count, 2] += drop_height

        velocities = np.zeros_like(positions)
        velocities[:drop_count, 2] = -np.random.uniform(1.0, 2.0)
        return positions, velocities

    @staticmethod
    def wave(num_particles, wave_amplitude, wave_frequency):
        positions = []
        n = int(np.sqrt(num_particles / 5))

        for x in np.linspace(-2.5, 2.5, n):
            for y in np.linspace(-2.0, 2.0, n):
                z_base = wave_amplitude * np.sin(wave_frequency * x)
                for z in np.linspace(0.1, z_base + 0.8, 5):
                    if len(positions) < num_particles:
                        positions.append([x, y, z])

        positions = np.array(positions[:num_particles], dtype=np.float32)
        velocities = np.zeros_like(positions)

        for i in range(len(positions)):
            velocities[i, 0] = wave_amplitude * wave_frequency * np.cos(wave_frequency * positions[i, 0]) * 0.5

        return positions, velocities

    @staticmethod
    def pour(num_particles, pour_height, pour_angle):
        positions = []
        angle_rad = np.radians(pour_angle)

        for i in range(num_particles):
            t = i / num_particles
            x = -1.5 + t * 3.0 * np.cos(angle_rad)
            y = np.random.uniform(-0.1, 0.1)
            z = pour_height - t * 1.5 * np.sin(angle_rad)
            positions.append([x, y, max(z, 0.1)])

        positions = np.array(positions, dtype=np.float32)
        velocities = np.zeros_like(positions)
        velocities[:, 0] = np.cos(angle_rad) * 1.5
        velocities[:, 2] = -np.sin(angle_rad) * 1.5
        return positions, velocities

    @staticmethod
    def fountain(num_particles, fountain_strength):
        positions = []

        for i in range(num_particles):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.2)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.random.uniform(0.1, 1.5)
            positions.append([x, y, z])

        positions = np.array(positions, dtype=np.float32)
        velocities = np.zeros_like(positions)

        for i in range(len(positions)):
            angle = np.arctan2(positions[i, 1], positions[i, 0])
            velocities[i, 0] = np.cos(angle) * fountain_strength * 0.3
            velocities[i, 1] = np.sin(angle) * fountain_strength * 0.3
            velocities[i, 2] = fountain_strength

        return positions, velocities

    @staticmethod
    def multiple_drops(num_particles, num_drops, drop_height):
        positions = []
        particles_per_drop = num_particles // num_drops

        for i in range(num_drops):
            angle = 2 * np.pi * i / num_drops
            drop_x = np.cos(angle) * 1.0
            drop_y = np.sin(angle) * 1.0

            drop_particles = []
            while len(drop_particles) < particles_per_drop:
                point = np.random.uniform(-0.2, 0.2, 3)
                if np.linalg.norm(point) <= 0.2:
                    drop_particles.append(point)

            for point in drop_particles[:particles_per_drop]:
                positions.append([point[0] + drop_x, point[1] + drop_y, point[2] + drop_height])

        positions = np.array(positions[:num_particles], dtype=np.float32)
        velocities = np.zeros_like(positions)

        for i in range(len(positions)):
            direction = -positions[i, :2] / (np.linalg.norm(positions[i, :2]) + 1e-6)
            velocities[i, 0] = direction[0] * 1.0
            velocities[i, 1] = direction[1] * 1.0
            velocities[i, 2] = -1.5

        return positions, velocities


def generate_scenario(scenario_type, scenario_params, num_frames, output_dir, scenario_name):
    """Generate one scenario"""
    # Create scenario
    if scenario_type == "dam_break":
        positions, velocities = ScenarioGenerator.dam_break(**scenario_params)
    elif scenario_type == "splash":
        positions, velocities = ScenarioGenerator.splash(**scenario_params)
    elif scenario_type == "wave":
        positions, velocities = ScenarioGenerator.wave(**scenario_params)
    elif scenario_type == "pour":
        positions, velocities = ScenarioGenerator.pour(**scenario_params)
    elif scenario_type == "fountain":
        positions, velocities = ScenarioGenerator.fountain(**scenario_params)
    elif scenario_type == "multiple_drops":
        positions, velocities = ScenarioGenerator.multiple_drops(**scenario_params)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    # Create simulator with varying parameters
    simulator = FastSPHSimulator(
        particle_mass=0.02,
        smoothing_radius=np.random.uniform(0.25, 0.35),
        gas_constant=np.random.uniform(0.8, 1.2),
        viscosity=np.random.uniform(0.005, 0.02),
        gravity_scale=np.random.uniform(0.9, 1.1)
    )

    # Simulate and save
    output_path = output_dir / scenario_name
    output_path.mkdir(exist_ok=True, parents=True)

    for frame in range(num_frames):
        frame_path = output_path / f"frame_{frame:04d}.npz"
        np.savez_compressed(frame_path, positions=positions, velocities=velocities)
        positions, velocities = simulator.step(positions, velocities)

    return len(positions), num_frames


def main():
    """Generate optimized dataset - 300 scenarios"""
    output_dir = Path("datasets/flip_simulations")
    output_dir.mkdir(exist_ok=True, parents=True)

    scenarios = []

    # 50 Dam Breaks
    logger.info("Planning 50 dam break scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'dam_break',
            'params': {
                'num_particles': np.random.randint(700, 1100),
                'dam_height': np.random.uniform(1.8, 2.8),
                'dam_width': np.random.uniform(0.9, 1.4),
                'dam_position_x': np.random.uniform(-2.0, -1.0)
            },
            'num_frames': np.random.randint(45, 55),
            'name': f"dambreak_{i:03d}"
        })

    # 50 Splashes
    logger.info("Planning 50 splash scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'splash',
            'params': {
                'num_particles': np.random.randint(600, 900),
                'drop_height': np.random.uniform(2.5, 3.5),
                'pool_particles': np.random.randint(100, 250)
            },
            'num_frames': np.random.randint(40, 50),
            'name': f"splash_{i:03d}"
        })

    # 50 Waves
    logger.info("Planning 50 wave scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'wave',
            'params': {
                'num_particles': np.random.randint(900, 1200),
                'wave_amplitude': np.random.uniform(0.25, 0.45),
                'wave_frequency': np.random.uniform(1.5, 2.5)
            },
            'num_frames': np.random.randint(50, 60),
            'name': f"wave_{i:03d}"
        })

    # 50 Pours
    logger.info("Planning 50 pour scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'pour',
            'params': {
                'num_particles': np.random.randint(600, 850),
                'pour_height': np.random.uniform(2.8, 3.8),
                'pour_angle': np.random.uniform(35, 55)
            },
            'num_frames': np.random.randint(45, 52),
            'name': f"pour_{i:03d}"
        })

    # 50 Fountains
    logger.info("Planning 50 fountain scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'fountain',
            'params': {
                'num_particles': np.random.randint(700, 1000),
                'fountain_strength': np.random.uniform(2.5, 3.5)
            },
            'num_frames': np.random.randint(48, 58),
            'name': f"fountain_{i:03d}"
        })

    # 50 Multiple Drops
    logger.info("Planning 50 multiple drops scenarios...")
    for i in range(50):
        scenarios.append({
            'type': 'multiple_drops',
            'params': {
                'num_particles': np.random.randint(800, 1100),
                'num_drops': np.random.randint(3, 5),
                'drop_height': np.random.uniform(2.7, 3.3)
            },
            'num_frames': np.random.randint(45, 53),
            'name': f"multidrops_{i:03d}"
        })

    logger.info(f"Total scenarios planned: {len(scenarios)}")
    print(f"\nGenerating {len(scenarios)} high-quality scenarios...")

    # Generate
    total_particles = 0
    total_frames = 0

    for scenario in tqdm(scenarios, desc="Generating scenarios"):
        num_particles, num_frames = generate_scenario(
            scenario['type'],
            scenario['params'],
            scenario['num_frames'],
            output_dir,
            scenario['name']
        )
        total_particles += num_particles
        total_frames += num_frames

    # Save metadata
    metadata = {
        'total_scenarios': len(scenarios),
        'total_frames': total_frames,
        'avg_particles': total_particles // len(scenarios),
        'scenario_types': {
            'dam_break': 50,
            'splash': 50,
            'wave': 50,
            'pour': 50,
            'fountain': 50,
            'multiple_drops': 50
        },
        'generation_date': datetime.now().isoformat(),
        'log_file': str(log_file)
    }

    metadata_path = output_dir / "dataset_metadata_optimized.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("="*60)
    logger.info("OPTIMIZED DATA GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total scenarios: {len(scenarios)}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Average particles: {total_particles // len(scenarios)}")
    logger.info(f"Training sequences: {total_frames - len(scenarios)}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info("="*60)

    print("\n" + "="*60)
    print("[SUCCESS] Optimized SPH Data Generation Complete!")
    print("="*60)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Total frames: {total_frames}")
    print(f"Training sequences: {total_frames - len(scenarios)}")
    print(f"Avg particles: {total_particles // len(scenarios)}")
    print(f"Metadata: {metadata_path}")
    print("="*60)


if __name__ == "__main__":
    main()
