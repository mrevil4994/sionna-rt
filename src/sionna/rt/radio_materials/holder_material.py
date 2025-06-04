#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mitsuba BSDF that holds the radio material used by a scene object"""

import mitsuba as mi

from .radio_material_base import RadioMaterialBase


class HolderMaterial(mi.BSDF):
    # pylint: disable=line-too-long
    r"""
    Class that holds the radio material used by a scene object

    Every scene object is hooked to an instance of this class which holds
    the radio material used for simulation. This enables changing the radio
    material by setting the held radio material.

    Note that when a scene is loaded, a holder is instantiated for each object
    (or group of merged objects) and attached to it.

    :param props: Properties that should be either empty or only store an instance of :class:`~sionna.rt.RadioMaterialBase` to be used as radio material by the scene object
    """

    def __init__(self, props : mi.Properties):
        super().__init__(props)

        # If there is an inner radio material, it is loaded as the held radio
        # material
        radio_material = None
        for k in props.property_names():
            v = props[k]
            if isinstance(v, RadioMaterialBase):
                if radio_material is None:
                    radio_material = v
                else:
                    raise ValueError("HolderMaterial only allows one nested"
                                    " radio material, but found several.")
            else:
                raise ValueError("HolderMaterial only allows one nested radio"
                                 f" material but found property \"{k}\" of type"
                                 f" {type(v)}.")
        self.radio_material = radio_material

        # Set the velocity vector
        self._velocity = mi.Vector3f(0, 0, 0)

    @property
    def radio_material(self):
        """
        Get/set the held radio material

        :type: :class:`~sionna.rt.RadioMaterialBase`
        """
        return self._radio_material

    @radio_material.setter
    def radio_material(self, radio_material):
        if radio_material is None:
            self._radio_material = None
            return
        if not isinstance(radio_material, RadioMaterialBase):
            raise ValueError("The radio material should be an instance of"
                             " RadioMaterialBase")
        self._radio_material = radio_material
        self.m_flags = radio_material.m_flags
        self.m_components = radio_material.m_components

    def to_string(self):
        return f"HolderMaterial[{self._radio_material}]"

    @property
    def velocity(self):
        """
        Get/set the velocity of the object attached to this holder [m/s]

        :type: :py:class:`mi.Vector3f`
        """
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        self._velocity = mi.Vector3f(v)

    # --- Forward all other methods to the underlying radio material
    def sample(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.sample(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def pdf(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.pdf(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_pdf(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval_pdf(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_diffuse_reflectance(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval_diffuse_reflectance(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_null_transmission(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval_null_transmission(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def has_attribute(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.has_attribute(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_attribute(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval_attribute(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_attribute_1(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.eval_attribute_1(*args, **kwargs)
        raise ValueError("No radio material attached to this holder")
    def eval_attribute_3(self, name, si, active=True):
        if name == "velocity":
            return self._velocity
        if self.radio_material is not None:
            return self.radio_material.eval_attribute_3(name, si, active)
        raise ValueError("No radio material attached to this holder")
    def traverse(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.traverse(*args, **kwargs)
    def parameters_changed(self, *args, **kwargs):
        if self.radio_material is not None:
            return self.radio_material.parameters_changed(*args, **kwargs)

mi.register_bsdf("holder-material", lambda props: HolderMaterial(props=props))
