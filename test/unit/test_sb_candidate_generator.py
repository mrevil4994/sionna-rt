#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import mitsuba as mi
import numpy as np

from sionna import rt
from sionna.rt import load_scene, ITURadioMaterial, InteractionType,\
    INVALID_SHAPE, INVALID_PRIMITIVE
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator


############################################################
# Utilities
############################################################

def load_box_scene(material_name, thickness, scattering_coefficient):

    scene = load_scene(rt.scene.box, merge_shapes=False)
    box = scene.get("box")

    box.radio_material = ITURadioMaterial(name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

def load_box_one_screen_scene(material_name, thickness, scattering_coefficient):

    scene = load_scene(rt.scene.box_one_screen, merge_shapes=False)
    screen = scene.get("screen")

    screen.radio_material = ITURadioMaterial(
                                name=f"mat-{material_name}",
                                itu_type=material_name,
                                thickness=thickness,
                                scattering_coefficient=scattering_coefficient)

    return scene

############################################################
# Unit tests
############################################################

@pytest.mark.parametrize("int_type_str", [
    'specular', # Specular reflection
    'transmission', # Transmission
])
def test_specular_reflection_transmission_depth_1(int_type_str):
    """
    Tests chains of depth 1 with specular or transmission

    Input
    ------
    type : str, 'specular' or 'transmission'
        'specular': Test with only specular reflection
        'transmission': Test with only transmission

    """

    assert int_type_str in ('specular', 'transmission'), "Wrong interaction type"


    if int_type_str == 'specular':
        thickness = 1.0 # Only reflection as using metal as material
        int_type = InteractionType.SPECULAR
    elif int_type_str == 'transmission':
        thickness = 0.0 # Only transmission
        int_type = InteractionType.REFRACTION

    source = mi.Point3f(0., 0., 1.5)
    target = mi.Point3f(1., 1., 1.)

    max_depth = 1
    samples_per_src = 100
    max_num_paths = 1000

    scene = load_box_scene("metal", thickness, scattering_coefficient=0.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()

    # Depth should be set to 1 and max_num_path to 13
    assert paths.buffer_size == 13
    assert paths.max_depth == 1

    # There should be one LoS, and all other interactions must be of the
    # selected type
    int_types_u, int_types_i, int_types_c = np.unique(int_types, return_counts=True, return_index=True)
    assert len(int_types_u) == 2
    assert InteractionType.NONE in int_types_u
    assert int_type in int_types_u
    assert 1 in int_types_c
    assert 12 in int_types_c

    # Index of the LoS
    if int_types_c[0] == 1:
        los_index = int_types_i[0]
    else:
        los_index = int_types_i[1]

    # Only one shape
    assert shapes[los_index] == INVALID_SHAPE
    assert np.unique(shapes).shape[0] == 2

    # 12 unique primitives should be found
    assert primitives[los_index] == INVALID_PRIMITIVE
    assert np.unique(primitives).shape[0] == 13

    # No paths should be valid, i.e., only candidates
    assert valid[los_index]
    assert np.all(np.logical_not(np.delete(valid, los_index)))

@pytest.mark.parametrize("int_type_str", [
    'specular', # Specular reflection
    'transmission', # Transmission
])
def test_specular_or_transmission_depth_1_multilink(int_type_str):
    """
    Tests chains of depth 1 with specular *or* transmission with multiple
    sources and targets

    Input
    ------
    type : str, 'specular' or 'transmission'
        'specular': Test with only specular reflection
        'transmission': Test with only transmission
    """

    assert int_type_str in ('specular', 'transmission'), "Wrong interaction type"

    if int_type_str == 'specular':
        thickness = 1.0 # Only reflection as using metal as material
        int_type = InteractionType.SPECULAR
    elif int_type_str == 'transmission':
        thickness = 0.0 # Only transmission
        int_type = InteractionType.REFRACTION

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 1
    samples_per_src = 100
    max_num_paths = 10000

    scene = load_box_scene("metal", thickness, scattering_coefficient=0.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to 1 and max_num_path to 12*6, as they are 12
    # primitives in the scene.mi_scene, and each of the 6 link should have the 12
    # primitives as candidates
    assert paths.buffer_size == 13*6 # 13 per link
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 6 # 1 per link

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)
    assert np.unique(los_src_indices).shape[0] == 2
    assert np.unique(los_tgt_indices).shape[0] == 3

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)
    src_indices = np.delete(src_indices, los_indices)
    tgt_indices = np.delete(tgt_indices, los_indices)

    # All interactions must be of the selected type
    assert np.unique(int_types) == np.array([int_type])

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # 12 unique primitives should be found for each (source, target) link
    primitives_per_link = {}
    for src_ind, tgt_ind, prim_ind in zip(src_indices, tgt_indices, primitives):
        key = (src_ind, tgt_ind)
        if key not in primitives_per_link:
            primitives_per_link[key] = []
        primitives_per_link[key].append(prim_ind)

    # There should be 6 links
    assert len(primitives_per_link) == 6
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            # There should be 6 unique primitives for each link
            assert np.unique(primitives_per_link[key]).shape[0] == 12

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

def test_specular_and_transmission_depth_1():
    """
    Test single reflection (depth of 1) with both specular reflection
    and transmission
    """

    source = mi.Point3f(0., 0., 2.5)
    target = mi.Point3f(1., 1., 1.)

    max_depth = 1
    samples_per_src = 1000
    max_num_paths = 10000

    # Scattering coefficient is set to 0
    # Set material to glass with a thickness of 1cm, which lead to almost equal
    # splitting of the energy between transmission and reflection
    scene = load_box_scene("glass", 0.01, scattering_coefficient=0.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()

    # Depth should be set to 1 and max_num_path to 24
    # Indeed, each primitive should have one transmitted path and
    # one specularly reflected path
    assert paths.buffer_size == 25
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 1 # Only a single LoS

    los_shape = shapes[los_indices]
    los_primitive = primitives[los_indices]
    los_valid = valid[los_indices]

    assert los_shape == INVALID_SHAPE
    assert los_primitive == INVALID_PRIMITIVE
    assert los_valid

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)

    # All interactions must be either specular reflection or transmission
    for int_type in int_types:
        assert int_type in (InteractionType.SPECULAR, InteractionType.REFRACTION)

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check that there is no redundancy
    inter_pair = np.stack([primitives, int_types], axis=1)
    inter_pair = np.unique(inter_pair, axis=0)
    assert inter_pair.shape[0] == 24

def test_specular_and_transmission_depth_1_multilink():
    """
    Test single reflection (depth of 1) with both specular reflection
    and transmission with multiple sources and targets
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 1
    samples_per_src = int(1e5)
    max_num_paths = int(1e4)

    # Scattering coefficient is set to 0
    # Set material to glass with a thickness of 1cm, which lead to almost equal
    # splitting of the energy between transmission and reflection
    scene = load_box_scene("glass", 0.01, scattering_coefficient=0.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()[:,0]
    primitives = paths.primitives.numpy()[:,0]
    int_types = paths.interaction_types.numpy()[:,0]
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to 1 and max_num_path to 25*6
    # Indeed, each link should have one transmitted path and
    # one specularly reflected path for each primitive
    assert paths.buffer_size == 25*6
    assert paths.max_depth == 1

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 6 # 1 per link

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)
    assert np.unique(los_src_indices).shape[0] == 2
    assert np.unique(los_tgt_indices).shape[0] == 3

    # Check NLoS paths

    shapes = np.delete(shapes, los_indices)
    primitives = np.delete(primitives, los_indices)
    valid = np.delete(valid, los_indices)
    int_types = np.delete(int_types, los_indices)
    src_indices = np.delete(src_indices, los_indices)
    tgt_indices = np.delete(tgt_indices, los_indices)

    # All interactions must be either specular reflection or transmission
    for int_type in int_types:
        assert int_type in (InteractionType.SPECULAR, InteractionType.REFRACTION)

    # Only one shape
    assert np.unique(shapes).shape[0] == 1

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check that there is no redundancy
    inter_pair = np.stack([src_indices, tgt_indices, primitives, int_types], axis=1)
    inter_pair = np.unique(inter_pair, axis=0)
    assert inter_pair.shape[0] == 24*6

def test_los_with_obstruction_multilink():
    r"""
    In the box scene with a screen and multiple links, check that LoS that should
    be obstructed are
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [4., -4.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 3., 3.],
                         [4., 0., -4.],
                         [2.5, 1.5, 2.5])

    max_depth = 1
    samples_per_src = int(1e3)
    max_num_paths = int(1e6)

    # Set material of the screen to glass with a thickness of 1cm, which lead to
    # almost equal splitting of the energy between transmission and reflection
    # Scattering coefficient for the screen set to 0
    scene = load_box_one_screen_scene("glass", 0.01, 0.0)

    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=True, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()
    primitives = paths.primitives.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Extract LoS and check it

    los_indices = np.where(int_types == InteractionType.NONE)[0]
    assert los_indices.shape[0] == 2

    los_shapes = shapes[los_indices]
    los_primitives = primitives[los_indices]
    los_valid = valid[los_indices]
    los_src_indices = src_indices[los_indices]
    los_tgt_indices = tgt_indices[los_indices]

    assert np.unique(los_shapes) == np.array([INVALID_SHAPE])
    assert np.unique(los_primitives) == np.array([INVALID_PRIMITIVE])
    assert np.all(los_valid)

    # There should be only 2 LoS:
    #   source 0 --> target 0
    #   source 1 --> target 2
    assert los_src_indices[0] == 0 and los_tgt_indices[0] == 0
    assert los_src_indices[1] == 1 and los_tgt_indices[1] == 2

def test_transmission_specular_high_depth():
    r"""
    Test chains that consists of specular and transmission only and of high
    depth
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 3., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e3)
    max_num_paths = int(1e6)

    # Set material of the screen to glass with a thickness of 1cm, which lead to
    # almost equal splitting of the energy between transmission and reflection
    # Scattering coefficient for the screen set to 0
    scene = load_box_one_screen_scene("glass", 0.01, 0.0)

    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    shapes = paths.shapes.numpy()
    primitives = paths.primitives.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Check interaction types
    num_paths = paths.buffer_size
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)
        transmission = np.equal(inters, InteractionType.REFRACTION)
        no_int = np.equal(inters, InteractionType.NONE)

        # Interaction type is specular or none
        assert np.all(np.any([specular, transmission, no_int]))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, np.logical_or(specular,
                                                                  transmission))

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check there are no duplicate candidates
    # First, aggregate all interaction for each link
    link_interactions = {}
    i = 1
    for src_ind, tgt_ind, shape_ind, prim_ind, int_type in zip(src_indices, tgt_indices, shapes, primitives, int_types):
        key = (src_ind, tgt_ind)
        if key not in link_interactions:
            link_interactions[key] = []
        inter = np.stack([shape_ind, prim_ind, int_type], axis=1)
        i += 1
        link_interactions[key].append(inter)
    # Check each link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            inter = link_interactions[key]
            inter = np.stack(inter, axis=0)
            inter = np.reshape(inter, [inter.shape[0], -1])
            _, counts = np.unique(inter, axis=0, return_counts=True)
            assert np.all(np.equal(counts, 1))

def test_specular_depth_high():
    """
    Test specular chains of high depth
    """

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e5)
    max_num_paths = int(1e7)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=0.)

    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    primitives = paths.primitives.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Check interaction types
    num_paths = paths.buffer_size
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)
        no_int = np.equal(inters, InteractionType.NONE)

        # Interaction type is specular or none
        assert np.all(np.logical_or(specular, no_int))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, specular)

    # No paths should be valid, i.e., only candidates
    assert np.all(np.logical_not(valid))

    # Check there are no duplicate candidates
    # First, aggregate all interaction for each link
    primitives_seq = {}
    int_types_seq = {}
    i = 1
    for src_ind, tgt_ind, prim_ind, int_type in zip(src_indices, tgt_indices, primitives, int_types):
        key = (src_ind, tgt_ind)
        if key not in primitives_seq:
            primitives_seq[key] = []
            int_types_seq[key] = []
        i += 1
        primitives_seq[key].append(prim_ind)
        int_types_seq[key].append(int_type)
    # Check each link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)

            prim_ind = primitives_seq[key]
            prim_ind = np.stack(prim_ind, axis=0)
            _, counts = np.unique(prim_ind, axis=0, return_counts=True)
            assert np.all(np.equal(counts, 1))

            # Check paths with depth 1
            int_type = int_types_seq[key]
            int_type = np.stack(int_type, axis=0)
            depth_1_paths_indices = np.where(int_type[:,1] == InteractionType.NONE)[0]
            depth_1_paths_prims = prim_ind[depth_1_paths_indices,0]
            # Should be 12 paths
            # Disabled as due to scheduling of threads, all first-order
            # specular paths might not be found when the path buffer is not
            # large enough in such scenarios.
            # assert depth_1_paths_prims.shape[0] == 12
            #
            _, prims_counts = np.unique(depth_1_paths_prims, return_counts=True)
            assert np.all(np.equal(prims_counts, 1))

def test_specular_prefixes():
    """
    Test that, for specular chains, all prefixes are listed as candidates
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 3
    samples_per_src = 100
    max_num_paths = int(1e5)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=0.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    primitives = paths.primitives.numpy()
    max_depth = paths.max_depth
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Test that all prefixes are listed as candidates
    # First, aggregate all interaction for each link
    link_interactions = {}
    for src_ind, tgt_ind, prim_ind in zip(src_indices, tgt_indices, primitives):
        key = (src_ind, tgt_ind)
        if key not in link_interactions:
            link_interactions[key] = []
        link_interactions[key].append(prim_ind)
    # Check the prefixes for every link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            prim_ind = link_interactions[key]
            prim_ind = np.stack(prim_ind, axis=0)

            num_paths = prim_ind.shape[0]
            for i in range(num_paths):
                # Extract the sequence of primitives forming this path
                path = prim_ind[i]

                # Compute the path depth
                for d in range(max_depth):
                    if path[d] == INVALID_PRIMITIVE:
                        break
                # Nothing to check if depth is 1
                if d == 1:
                    continue

                # Remove last interaction to get a prefix
                prefix = path[:d-1]
                prefix = np.pad(prefix, [[0,max_depth-d+1]],
                                constant_values=INVALID_PRIMITIVE)

                # Check that the prefix is part of the found paths
                found = False
                for j in range(num_paths):
                    path_2 = prim_ind[j]
                    eq = np.sum(np.abs(path_2-prefix))
                    found = np.equal(eq, 0.)
                    if found:
                        break
                assert found

def test_diffuse_depth_high():
    """
    Test paths made only of diffuse reflection
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 10
    samples_per_src = int(1e4)
    max_num_paths = int(1e6)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=1.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    primitives = paths.primitives.numpy()
    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Number of paths should be max_depth*samples_per_src*num_links
    # if not constraints by max_num_paths
    assert paths.buffer_size == samples_per_src*max_depth*6

    # Check interaction types
    num_paths = paths.buffer_size
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        diffuse = np.equal(inters, InteractionType.DIFFUSE)
        no_int = np.equal(inters, InteractionType.NONE)

        # Interaction type is diffuse or none
        assert np.all(np.logical_or(diffuse, no_int))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, diffuse)

    # All paths should be valid, i.e., no candidates
    assert np.all(valid)

def test_diffuse_prefixes():
    """
    Check that, in the case of the box scene for which there is no occlusion,
    all all prefixes are listed as valid paths
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    max_depth = 3
    samples_per_src = int(100)
    max_num_paths = int(1e4)

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=1.)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    primitives = paths.primitives.numpy()
    max_depth = paths.max_depth
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Test that all prefixes are listed as candidates
    # First, aggregate all interaction for each link
    link_interactions = {}
    for src_ind, tgt_ind, prim_ind in zip(src_indices, tgt_indices, primitives):
        key = (src_ind, tgt_ind)
        if key not in link_interactions:
            link_interactions[key] = []
        link_interactions[key].append(prim_ind)
    # Check the prefixes for every link
    for src_ind in range(2):
        for tgt_ind in range(3):
            key = (src_ind, tgt_ind)
            primitives = link_interactions[key]
            primitives = np.stack(primitives, axis=0)

            num_paths = primitives.shape[0]
            for i in range(num_paths):
                # Extract the sequence of primitives forming this path
                path = primitives[i]

                # Compute the path depth
                for d in range(max_depth):
                    if path[d] == INVALID_PRIMITIVE:
                        break
                d = d+1
                # Nothing to check if depth is 1
                if d == 1:
                    continue

                # Remove last interaction to get a prefix
                prefix = path[:d-1]
                prefix = np.pad(prefix, [[0,max_depth-d+1]],
                                constant_values=INVALID_PRIMITIVE)

                # Check that the prefix is part of the found paths
                found = False
                for j in range(num_paths):
                    path_2 = primitives[j]
                    eq = np.sum(np.abs(path_2-prefix))
                    found = np.equal(eq, 0.)
                    if found:
                        break
                assert found

def test_diffuse_specular():
    """
    Check paths made of mixtures of specular and diffuse
    """
    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    # To compute correctly the probability of a diffuse or reflection event,
    # we need to average over the path depth to not be biased by the fact
    # that specular paths are not duplicated.
    max_depth = 200
    samples_per_src = int(1e2)
    max_num_paths = int(1e5)

    # Probabilty of an interaction to be specular
    ps = 0.7

    # Set material to metal which leads to all the energy being reflected
    # Scattering coefficient set to 0
    scene = load_box_scene("metal", 0.01, scattering_coefficient=np.sqrt(1.-ps))
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, sources, targets, samples_per_src, max_num_paths, max_depth,
                   los=False, refraction=True, specular_reflection=True,
                   diffuse_reflection=True, seed=1)
    paths.shrink()

    int_types = paths.interaction_types.numpy()
    valid = paths.valid.numpy()
    num_paths = paths.buffer_size

    # Depth should be set to `max_depth`
    assert paths.max_depth == max_depth

    # Check interaction types
    paths_active = np.full([num_paths], True)
    for d in range(max_depth):
        inters = int_types[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)
        diffuse = np.equal(inters, InteractionType.DIFFUSE)
        no_int = np.equal(inters, InteractionType.NONE)
        bounce = np.logical_or(specular,diffuse)

        # The ray bounced or the interaction is none
        assert np.all(np.logical_or(bounce, no_int))

        # If paths is done, there should be only none interaction
        assert np.all(np.logical_or(paths_active, no_int))

        # Update paths state
        paths_active = np.logical_and(paths_active, bounce)

    # Check that the ratio of specular paths matches the configured
    # scattering coefficient.
    # Only the paths with depth = max_depth are used to get an unbiased
    # estimate
    long_paths_ind = np.where(paths_active)[0]
    long_types_chains = int_types[long_paths_ind]
    num_specular = 0
    for d in range(max_depth):
        inters = long_types_chains[:,d]
        specular = np.equal(inters, InteractionType.SPECULAR)

        num_specular += np.sum(specular)
    ratio_specular = num_specular/long_types_chains.shape[0]\
                                 /long_types_chains.shape[1]
    assert np.abs(ratio_specular-ps) < 0.01

    # Ensure that paths ending by a diffuse reflection are valid, whereas those
    # ending by a specular reflection are not
    for d in range(max_depth):
        specular = np.equal(int_types[:,d], InteractionType.SPECULAR)
        diffuse = np.equal(int_types[:,d], InteractionType.DIFFUSE)
        bounce = np.logical_or(specular,diffuse)

        # Indices of paths for which the previous bounce was
        # the last one
        if d == max_depth-1:
            last_bounce = bounce
        else:
            next_is_none = np.equal(int_types[:,d+1], InteractionType.NONE)
            last_bounce = np.logical_and(bounce, next_is_none)

        diff_ind = np.where(np.logical_and(diffuse,last_bounce))[0]
        spec_ind = np.where(np.logical_and(specular,last_bounce))[0]

        assert np.all(valid[diff_ind])
        assert np.all(np.logical_not(valid[spec_ind]))

@pytest.mark.parametrize("los", [True, False])
@pytest.mark.parametrize("refraction", [True, False])
@pytest.mark.parametrize("specular_reflection", [True, False])
@pytest.mark.parametrize("diffuse_reflection", [True, False])
def test_intertaction_type_flags(los, refraction, specular_reflection,
                                 diffuse_reflection):

    source = mi.Point3f(0., 0., 1.5)
    target = mi.Point3f(1., 1., 1.)

    max_depth = 1
    samples_per_src = 100
    max_num_paths = 1000

    scene = load_box_scene("glass", 0.01, 0.2)
    tracer = SBCandidateGenerator()

    paths = tracer(scene.mi_scene, source, target, samples_per_src, max_num_paths, max_depth,
                los=los, refraction=refraction, specular_reflection=specular_reflection,
                diffuse_reflection=diffuse_reflection, seed=1)
    paths.shrink()
    interactions = paths.interaction_types.numpy()
    interactions = np.squeeze(interactions)

    has_los = InteractionType.NONE in interactions
    assert np.logical_not(np.bitwise_xor(los, has_los))

    has_specular = InteractionType.SPECULAR in interactions
    assert np.logical_not(np.bitwise_xor(specular_reflection, has_specular))

    has_refraction = InteractionType.REFRACTION in interactions
    assert np.logical_not(np.bitwise_xor(refraction, has_refraction))

    has_diffuse = InteractionType.DIFFUSE in interactions
    assert np.logical_not(np.bitwise_xor(diffuse_reflection, has_diffuse))
