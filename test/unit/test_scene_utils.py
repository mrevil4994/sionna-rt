#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
from os.path import join
import tempfile

import pytest

import mitsuba as mi
import drjit as dr
from sionna import rt
from sionna.rt import load_scene, SceneObject, HolderMaterial,  RadioMaterial,\
    ITURadioMaterial


def test01_scene_preprocessing():
    scene_processed = load_scene(rt.scene.box_two_screens)
    scene_processed = scene_processed.mi_scene

    # Check that all BSDFs in the scene were correctly replaced by our custom radio BSDF.
    for sh in scene_processed.shapes():
        assert isinstance(sh.bsdf(), HolderMaterial)
        assert isinstance(sh.bsdf().radio_material, RadioMaterial)


def test02_merge_exclude_regex():
    tmp_path = join(tempfile.gettempdir(), "test_scene.xml")

    with open(tmp_path, "w") as f:
        f.write("""<scene version="2.1.0">
    <bsdf type="diffuse" id="bsdf-1"/>

    <shape type="cube" id="floor">
        <ref name="bsdf" id="bsdf-1"/>
    </shape>
    <shape type="cube" id="ceiling">
        <ref name="bsdf" id="bsdf-1"/>
    </shape>

    <shape type="cube" id="car-1">
        <ref name="bsdf" id="bsdf-1"/>
    </shape>
    <shape type="cube" id="car-2">
        <ref name="bsdf" id="bsdf-1"/>
    </shape>
    <shape type="cube" id="car-3">
        <ref name="bsdf" id="bsdf-1"/>
    </shape>
</scene>""")


    scene_processed = load_scene(tmp_path,
        merge_shapes_exclude_regex=r"^car-.+")
    scene_processed = scene_processed.mi_scene

    # 1 merged shape + 3 car shapes
    assert len(scene_processed.shapes()) == 4
    for shape in scene_processed.shapes():
        id = shape.id()
        assert id == "merged-shapes" or id.startswith("car-")

    os.remove(tmp_path)

def test03_scene_add_remove():
    tmp_path = join(tempfile.gettempdir(), "test_scene.xml")

    with open(tmp_path, "w") as f:
            f.write("""
        <scene version="2.1.0">

            <emitter type="constant"/>

            <integrator type="path"/>

            <bsdf type="diffuse" id="bsdf1"/>

            <shape type="cube" id="shape1">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

            <shape type="cube" id="shape2">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

        </scene>""")

    scene = load_scene(tmp_path, merge_shapes=False)
    shape_rm = scene.objects["shape1"].radio_material
    original_mi_scene = scene.mi_scene
    scene.edit()
    edited1_mi_scene = scene.mi_scene

    # 1. No change: everything should be preserved
    assert not edited1_mi_scene.sensors()
    assert edited1_mi_scene.environment() == original_mi_scene.environment()
    assert edited1_mi_scene.integrator() == original_mi_scene.integrator()
    assert len(edited1_mi_scene.emitters()) == 1  # Just the envmap
    assert len(edited1_mi_scene.shapes()) == 2
    assert set(s.id() for s in edited1_mi_scene.shapes()) == {"shape1", "shape2"}
    for s1, s2 in zip(original_mi_scene.shapes(), edited1_mi_scene.shapes()):
        assert s1 == s2

    # 2. Add some shapes and remove some other
    car_rm = ITURadioMaterial("car-mat", "metal", 0.01)
    cars = [
        SceneObject(fname=rt.scene.low_poly_car,
                    name="car1",
                    radio_material=car_rm),
        SceneObject(fname=rt.scene.low_poly_car,
                    name="car2",
                    radio_material=scene.radio_materials["bsdf1"])
        ]

    # Scene is edited in-place
    scene.edit(add=cars, remove=["shape1"])
    edited2_mi_scene = scene.mi_scene
    assert not edited2_mi_scene.sensors()
    assert edited2_mi_scene.environment() == original_mi_scene.environment()
    assert edited2_mi_scene.integrator() == original_mi_scene.integrator()
    assert len(edited2_mi_scene.emitters()) == 1  # Just the envmap
    assert len(scene.objects) == (2 - 1) + 2
    assert set(o.name for o in scene.objects.values()) == {"shape2", "car1", "car2"}

    for obj in scene.objects.values():
        # All shapes use the main BSDF except "car1"
        if obj.name == "car1":
            assert obj.radio_material is car_rm
        else:
            assert obj.radio_material is shape_rm


    # 3. Remove some shape that we added earlier
    scene.edit(remove=cars[0])
    assert len(scene.objects) == 2
    assert set(o.name for o in scene.objects.values()) == {"shape2", "car2"}

    # 4. Add a shape with an existing ID
    new_car = SceneObject(fname=rt.scene.low_poly_car,
                          name="car2",
                          radio_material=car_rm)
    with pytest.raises(ValueError, match=r"this ID is already used in the scene"):
        scene.edit(add=new_car)

    os.remove(tmp_path)

def test04_scene_object_scaling():
    tmp_path = join(tempfile.gettempdir(), "test_scene.xml")

    with open(tmp_path, "w") as f:
        f.write("""
        <scene version="2.1.0">

            <emitter type="constant"/>

            <integrator type="path"/>

            <bsdf type="diffuse" id="bsdf1"/>

            <shape type="cube" id="shape1">
                <ref name="bsdf" id="bsdf1"/>
            </shape>

        </scene>""")

    scene = load_scene(tmp_path, merge_shapes=False)
    cube = scene.objects["shape1"]

    # Helper functions
    def assert_bbox_is(min_should_be, max_should_be):
        shape_bb = cube._mi_shape.bbox()
        assert dr.allclose(shape_bb.min, min_should_be, atol=1e-5)
        assert dr.allclose(shape_bb.max, max_should_be, atol=1e-5)

    def reset_position():
        cube.position = mi.Point3f(0.0, 0.0, 0.0)  # Center the cube
        cube.look_at(mi.Point3f(1.0, 0.0, 0.0))  # Look in the positive x direction

    reset_position()

    # Sanity check the box bounds before scaling
    assert_bbox_is([-1, -1, -1], [1, 1, 1])
    assert dr.all(cube.scaling == mi.Vector3f(1.0))

    # Scale by a scalar value
    scalar = 10
    cube.scaling = scalar
    assert_bbox_is([-scalar] * 3, [scalar] * 3)
    assert dr.all(cube.scaling == mi.Vector3f(scalar))

    # Reassign scalar value
    scalar = 5
    cube.scaling = scalar
    assert_bbox_is([-scalar] * 3, [scalar] * 3)
    assert dr.all(cube.scaling == mi.Vector3f(scalar))

    # Negative scaling fails
    with pytest.raises(ValueError, match=r"Scaling must be positive"):
        cube.scaling = -1

    # Scale by a vector
    new_scale = mi.Vector3f(2.0, 4.0, 6.0)
    cube.scaling = new_scale
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])
    assert dr.all(cube.scaling == new_scale)

    # Reassign vector value
    new_scale = mi.Vector3f(1.2, 2.3, 3.4)
    cube.scaling = new_scale
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])
    assert dr.all(cube.scaling == new_scale)

    # Negative scaling fails
    with pytest.raises(ValueError, match=r"Scaling must be positive"):
        cube.scaling = mi.Vector3f(-1.0, 1.0, 1.0)

    # Translated and rotated cube scales correctly
    reset_position()
    cube.position = mi.Point3f(3.0, 6.0, 9.0) # Translate somewhere
    cube.look_at(mi.Point3f(-1.0, -1.0, -1.0)) # Rotate it 

    new_scale = mi.Vector3f(10.0, 5.0, 15.0) # Scale
    cube.scaling = new_scale

    reset_position() # After resetting position the bbox should be as below
    assert_bbox_is([-new_scale.x, -new_scale.y, -new_scale.z], [new_scale.x, new_scale.y, new_scale.z])

    # Translation and rotation is unaffected by scaling
    reset_position()
    cube.scaling = mi.Vector3f(1.0) # Reset scaling
    cube.position = mi.Point3f(2.0, 4.0, 6.0) # Translate somewhere
    cube.look_at(mi.Point3f(1.0, 1.0, 1.0)) # Rotate it 

    scene_params = cube._scene.mi_scene_params
    vp_key = cube._mi_shape.id() + ".vertex_positions"
    vertices_before_scaling = dr.unravel(mi.Point3f, scene_params[vp_key])

    cube.scaling = mi.Vector3f(1.23, 1.5, 7.0) # Scale it by some amount
    cube.scaling = mi.Vector3f(1.0) # Reset the scale

    # If translation and rotation unaffected then all vertices should be the same
    scene_params = cube._scene.mi_scene_params
    vp_key = cube._mi_shape.id() + ".vertex_positions"
    vertices_after_scaling = dr.unravel(mi.Point3f, scene_params[vp_key])
    assert dr.allclose(vertices_before_scaling, vertices_after_scaling, atol=1e-5)
