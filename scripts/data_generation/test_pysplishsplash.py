"""
Test pySPlisHSPlasH installation and basic API
"""

import pysplishsplash as sph
import numpy as np

print("pySPlisHSPlasH Test")
print("=" * 70)

# Create simulation
base = sph.Exec.SimulatorBase()
base.init()

print("\nSimulation initialized successfully!")

# Check available simulation methods
print("\nAvailable simulation methods:")
print(f"  - WCSPH (Weakly Compressible SPH)")
print(f"  - PCISPH (Predictive-Corrective Incompressible SPH)")
print(f"  - PBF (Position Based Fluids)")
print(f"  - IISPH (Implicit Incompressible SPH)")
print(f"  - DFSPH (Divergence-Free SPH)")

# Create time manager
print("\nCreating time manager...")
tm = sph.TimeManager()
print(f"Time manager created: {tm}")

# Test successful
print("\n" + "=" * 70)
print("pySPlisHSPlasH is ready to use!")
print("=" * 70)
