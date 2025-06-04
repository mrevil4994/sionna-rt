#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Meshes-related utilities.
"""

import os
import mitsuba as mi
import drjit as dr

import sionna
from .geometry import rotation_matrix


def clone_mesh(mesh : mi.Mesh, name: str | None = None,
               props: mi.Properties | None = None) -> mi.Mesh:
    """
    Clone a Mitsuba mesh object, preserving some of its important properties.

    :param name: Name (id) of the cloned object.
        If :py:class:`None`, the clone will be named as
        ``<original_name>-clone``.

    :param props: Pre-populated properties to used in the new Mitsuba mesh.
        Allows overriding the BSDF, emitter, etc.
    """
    # Clone name
    clone_name = name if (name is not None) else f"{mesh.id()}-clone"

    # Set ID and radio material for the clone
    if props is None:
        props = mi.Properties()
    props.set_id(clone_name)
    if "bsdf" not in props:
        props['bsdf'] = mi.load_dict({'type': 'holder-material'})
    if "flip_normals" not in props:
        # TODO: how can we preserve `flip_normals`?
        props['flip_normals'] = True

    # Instantiate clone mesh
    result = mi.Mesh(name=clone_name,
                     vertex_count=mesh.vertex_count(),
                     face_count=mesh.face_count(),
                     props=props,
                     has_vertex_normals=mesh.has_vertex_normals(),
                     has_vertex_texcoords=mesh.has_vertex_texcoords())

    # Copy mesh parameters
    params = mi.traverse(result)
    params["vertex_positions"] = mesh.vertex_positions_buffer()
    params["vertex_normals"] = mesh.vertex_normals_buffer()
    params["vertex_texcoords"] = mesh.vertex_texcoords_buffer()
    params["faces"] = mesh.faces_buffer()
    params.update()

    return result

def load_mesh(fname: str) -> mi.Mesh:
    # pylint: disable=line-too-long
    """
    Load a mesh from a file

    This function loads a mesh from a given file and returns it as a Mitsuba mesh.
    The file must be in either PLY or OBJ format.

    :param fname: Filename of the mesh to be loaded

    :return: Mitsuba mesh object representing the loaded mesh
    """

    mesh_type = os.path.splitext(fname)[1][1:]
    if mesh_type not in ('ply', 'obj'):
        raise ValueError("Invalid mesh type."
                            " Supported types: `ply` and `obj`")

    mi_mesh = mi.load_dict({'type': mesh_type,
                                'filename': fname,
                                'flip_normals': True,
                                'bsdf' : {'type': 'holder-material'}
                                })

    # We need to add a radio material to the object to enable shooting
    # and bouncing of rays
    mi_mesh.bsdf().radio_material = sionna.rt.RadioMaterial("dummy", 1.0)

    return mi_mesh

def transform_mesh(mesh : mi.Mesh,
                   translation : mi.Point3f | None = None,
                   rotation : mi.Point3f | None = None,
                   scale : mi.Point3f | None = None):
    # pylint: disable=line-too-long
    r"""
    In-place transformation of a mesh by applying translation, rotation, and scaling

    The order of the transformations is as follows:

    1. Scaling
    2. Rotation
    3. Translation

    Before applying the transformations, the mesh is centered.

    :param mesh: Mesh to be edited. The mesh is modified in-place.
    :param translation: Translation vector to apply
    :param rotation: Rotation angles [rad] specified through three angles
                     :math:`(\alpha, \beta, \gamma)` corresponding to a 3D rotation as defined in :eq:`rotation`
    :param scale: Scaling vector for scaling along the x, y, and z axes
    """

    params = mi.traverse(mesh)

    # Read vertex positions
    v = params["vertex_positions"]
    v = dr.unravel(mi.Point3f, v)

    # Center the mesh
    c = 0.5 * (mesh.bbox().min + mesh.bbox().max)
    v -= c

    # Scale the mesh
    if scale is not None:
        scale = mi.Point3f(scale)
        v *= scale

    # Rotate the mesh
    if rotation is not None:
        rotation = mi.Point3f(rotation)
        rot_matrix = rotation_matrix(rotation)
        v = rot_matrix @ v

    # Translate
    if translation is None:
        translation = c
    else:
        translation = mi.Point3f(translation)
        translation += c
    v += translation

    params["vertex_positions"] = dr.ravel(v)
    params.update()
