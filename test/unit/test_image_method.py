#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import mitsuba as mi
import numpy as np
import drjit as dr

import sionna.rt as rt
from sionna.rt import load_scene, InteractionType
from sionna.rt.path_solvers.sb_candidate_generator import SBCandidateGenerator
from sionna.rt.path_solvers.image_method import ImageMethod


############################################################
# Utilities
############################################################

def check_specular_path(source, target, int_types, vertices, normals):
    r"""
    Check that all the specular reflection and refraction in a specular chain
    are correctly reflected/refracted.
    """

    depth = np.where(int_types == InteractionType.NONE)[0]
    depth = depth[0] if depth.size > 0 else int_types.shape[0]

    int_types = int_types[:depth]
    vertices = vertices[:depth]
    normals = normals[:depth]

    # [depth + 2, 3]
    source = np.expand_dims(source, axis=0)
    target = np.expand_dims(target, axis=0)
    vertices = np.concatenate([source, vertices, target], axis=0)
    # Direction of the rays
    # [depth + 1, 3]
    directions = vertices[1:] - vertices[:-1]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    valid = True
    for d in range(depth):
        # Negative incoming ray direction
        in_dir = -directions[d]
        # Outgoing ray direction
        out_dir = directions[d+1]
        # Specular reflection
        if int_types[d] == InteractionType.SPECULAR:
            normal = normals[d]
            normal = normal*np.sign(np.dot(normal, in_dir))
            sum_in_out = in_dir + out_dir
            sum_in_out /= np.linalg.norm(sum_in_out, keepdims=True)
            cos_dis = 1. - np.dot(normal, sum_in_out)
            valid = np.isclose(cos_dis, 0., atol=1e-3)
        elif int_types[d] == InteractionType.REFRACTION:
            cos_dis = 1. - np.dot(-in_dir, out_dir)
            valid = np.isclose(cos_dis, 0., atol=1e-3)
        elif int_types[d] == InteractionType.DIFFUSE:
            valid = False # Should not happen
        if not valid:
            break

    return valid

def compute_face_normals(scene):

    def normal(p1, p2, p3):
        n = dr.cross(p2 - p1, p3 - p1)
        n = dr.normalize(n)
        return n

    face_normals = {}
    for shape in scene.mi_scene.shapes():
        shape_ind = dr.reinterpret_array(mi.UInt, mi.ShapePtr(shape))[0]
        face_normals[shape_ind] = {}
        face_count = shape.face_count()
        for i in range(face_count):
            vis = shape.face_indices(i)
            vps = [shape.vertex_position(vi) for vi in vis]
            n = normal(*vps).numpy()
            face_normals[shape_ind][i] = n[:,0]

    return face_normals

def ray_trace(scene, sources, targets, num_samples, max_num_paths, max_depth):

    cand_gen = SBCandidateGenerator()
    im = ImageMethod()

    paths = cand_gen(scene.mi_scene, sources, targets, num_samples,
                     max_num_paths, max_depth, los=False,
                     specular_reflection=True, diffuse_reflection=True,
                     refraction=True, seed=42)
    paths.shrink()
    paths = im(scene.mi_scene, paths, sources, targets)
    paths.discard_invalid()
    num_paths = paths.buffer_size

    # Interaction types
    int_types = paths.interaction_types.numpy()

    # Vertices
    vertices_x = paths.vertices_x.numpy()
    vertices_y = paths.vertices_y.numpy()
    vertices_z = paths.vertices_z.numpy()
    vertices = np.stack([vertices_x, vertices_y, vertices_z], axis=-1)

    # Paths sources and targets indices
    src_indices = paths.source_indices.numpy()
    tgt_indices = paths.target_indices.numpy()

    # Normals
    face_normals = compute_face_normals(scene)
    shapes_ind = paths.shapes.numpy()
    prims_ind = paths.primitives.numpy()
    normals = []
    for p in range(num_paths):
        normals.append([])
        for d in range(max_depth):
            si = shapes_ind[p][d]
            it = int_types[p][d]
            if it == InteractionType.NONE:
                n = np.array([0., 0., 0.])
            else:
                pi = prims_ind[p][d]
                n = face_normals[si][pi]
            normals[p].append(n)
    normals = np.array(normals)

    sources = sources.numpy()
    sources = np.transpose(sources, [1, 0])
    targets = targets.numpy()
    targets = np.transpose(targets, [1, 0])

    sources = np.take(sources, src_indices, axis=0)
    targets = np.take(targets, tgt_indices, axis=0)

    return sources, targets, int_types, vertices, normals

def ray_trace_box_scene(scattering_coefficient):

    scene = load_scene(rt.scene.box, merge_shapes=False)
    box = scene.get("box")

    box.radio_material = rt.ITURadioMaterial(
                            name="metal-mat",
                            itu_type="metal",
                            thickness=1.0,
                            scattering_coefficient=scattering_coefficient)

    # 2 sources
    sources = mi.Point3f([-3., -3],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([3., 0., 3.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    num_samples = int(1e5)
    max_num_paths = int(1e6)
    max_depth = 5

    return ray_trace(scene, sources, targets, num_samples, max_num_paths,
                     max_depth)

def ray_trace_box_two_screens_scene(scattering_coefficient_box):

    scene = load_scene(rt.scene.box_two_screens, merge_shapes=False)
    box = scene.get("box")
    screen_1 = scene.get("screen_1")
    screen_2 = scene.get("screen_2")

    box.radio_material = rt.ITURadioMaterial(
                            name="metal-mat",
                            itu_type="metal",
                            thickness=1.0,
                            scattering_coefficient=scattering_coefficient_box)

    screen_1.radio_material = rt.ITURadioMaterial(name="glass-mat",
                                                  itu_type="glass",
                                                  thickness=0.01,
                                                  scattering_coefficient=0.0)
    screen_2.radio_material = screen_1.radio_material

    # 2 sources
    sources = mi.Point3f([-4., -4],
                         [1., -1.],
                         [2.5, 2.5])

    # 3 targets
    targets = mi.Point3f([4., 0., 4.],
                         [1., 0., -1.],
                         [2.5, 1.5, 2.5])
    # 2x3 = 6 links

    num_samples = int(1e5)
    max_num_paths = int(1e6)
    max_depth = 5

    return ray_trace(scene, sources, targets, num_samples, max_num_paths,
                     max_depth)

############################################################
# Unit tests
############################################################

def test_specular_chains_specular_reflections():
    r"""
    Test paths consisting of specular reflections only
    """

    sources, targets, int_types, vertices, normals = ray_trace_box_scene(0.0)

    num_paths = int_types.shape[0]
    for p in range(num_paths):
        ts = int_types[p]
        vs = vertices[p]
        ns = normals[p]
        src = sources[p]
        tgt = targets[p]
        valid = check_specular_path(src, tgt, ts, vs, ns)
        assert valid

def test_specular_chains_specular_reflections_refractions():
    r"""
    Test paths consisting of specular reflections and refractions only
    """

    sources, targets, int_types, vertices, normals\
        = ray_trace_box_two_screens_scene(0.0)

    num_paths = int_types.shape[0]
    for p in range(num_paths):
        ts = int_types[p]
        vs = vertices[p]
        ns = normals[p]
        src = sources[p]
        tgt = targets[p]
        v = check_specular_path(src, tgt, ts, vs, ns)
        assert v

def test_specular_suffixes():
    r"""
    Test that specular suffixes are valid
    """

    sources, targets, int_types, vertices, normals\
        = ray_trace_box_two_screens_scene(np.sqrt(0.5))

    num_paths = int_types.shape[0]
    max_depth = int_types.shape[1]
    # Depth of the last diffuse reflection
    last_dr = -np.ones(num_paths, dtype='int')
    for p in range(num_paths):
        ts = int_types[p]
        for d in range(max_depth):
            if ts[d] == InteractionType.DIFFUSE:
                last_dr[p] = d

    # Keep only specular suffixes
    # Source of the specular suffix
    source_sf = sources
    for p in range(num_paths):
        if last_dr[p] == -1:
            continue # The path is a specular chain
        ldr = last_dr[p]
        source_sf[p] = vertices[p][ldr]
        d = 0
        for d_sc in range(ldr+1, max_depth):
            int_types[p][d] = int_types[p][d_sc]
            vertices[p][d] = vertices[p][d_sc]
            normals[p][d] = normals[p][d_sc]
            d += 1
        while d < max_depth:
            int_types[p][d] = InteractionType.NONE
            d += 1

    # Test specular suffixes
    for p in range(num_paths):
        ts = int_types[p]
        if ts[0] == InteractionType.NONE:
            continue # Skip empty suffixes
        vs = vertices[p]
        ns = normals[p]
        src = source_sf[p]
        tgt = targets[p]
        v = check_specular_path(src, tgt, ts, vs, ns)
        assert v
