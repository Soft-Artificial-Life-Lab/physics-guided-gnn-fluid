"""
Blender FLIP Fluid Simulation with Rigid Body
Simple test to verify Blender FLIP works
"""

import bpy
import os
import sys

# Clear existing scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Domain size
domain_size = 2.0

# Create fluid domain
bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(domain_size/2, domain_size/2, domain_size/2))
domain = bpy.context.object
domain.name = "FluidDomain"

# Add fluid simulation
bpy.ops.object.modifier_add(type='FLUID')
fluid_mod = domain.modifiers["Fluid"]
fluid_mod.fluid_type = 'DOMAIN'
domain_settings = fluid_mod.domain_settings
domain_settings.domain_type = 'LIQUID'
domain_settings.resolution_max = 64
domain_settings.simulation_method = 'FLIP'
domain_settings.use_mesh = False  # Particle output only
domain_settings.cache_directory = "//fluid_cache"

# Create fluid inflow (water block)
bpy.ops.mesh.primitive_cube_add(size=0.6, location=(domain_size/2, domain_size/2, domain_size*0.7))
fluid_source = bpy.context.object
fluid_source.name = "FluidSource"

# Add fluid flow
bpy.ops.object.modifier_add(type='FLUID')
fluid_source.modifiers["Fluid"].fluid_type = 'FLOW'
fluid_source.modifiers["Fluid"].flow_settings.flow_type = 'LIQUID'
fluid_source.modifiers["Fluid"].flow_settings.use_inflow = False  # Initial volume

# Create rigid body obstacle (sphere)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(domain_size/2, domain_size/2, domain_size*0.8))
obstacle = bpy.context.object
obstacle.name = "RigidSphere"

# Add effector (rigid body for fluid)
bpy.ops.object.modifier_add(type='FLUID')
obstacle.modifiers["Fluid"].fluid_type = 'EFFECTOR'
obstacle.modifiers["Fluid"].effector_settings.effector_type = 'COLLISION'

# Add rigid body physics
bpy.ops.rigidbody.object_add()
obstacle.rigid_body.type = 'ACTIVE'
obstacle.rigid_body.mass = 1.0

# Set frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 80

# Bake simulation
print("Baking fluid simulation...")
bpy.ops.fluid.bake_all()

print("FLIP simulation complete!")
print(f"Domain: {domain.name}")
print(f"Fluid source: {fluid_source.name}")
print(f"Rigid body: {obstacle.name}")
