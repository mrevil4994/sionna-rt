#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import mitsuba as mi
import numpy as np
import drjit as dr
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver


@pytest.mark.parametrize("phi",  np.linspace(-np.pi*0.5, np.pi*0.5, 10))
def test_synthetic_array(phi):
    """Test that synthetic and real arrays elad to similar channel impulse
    in the far field responses"""

    # Load empty scene
    scene = load_scene()
    wavelength = scene.frequency.numpy()[0]

    # Solver
    solver = PathSolver()

    # Set arrays
    scene.tx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="VH")
    scene.rx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="cross")

    d = 100.0
    # Transmitter and receiver
    x = d*dr.cos(phi)
    y = d*dr.sin(phi)
    rx = Receiver("rx", position=mi.Point3f(0., 0., 0.))
    tx = Transmitter("tx", position=mi.Point3f(x, y, 0))
    tx.look_at(rx)
    scene.add(tx)
    scene.add(rx)

    # Synthetic array
    paths = solver(scene, synthetic_array = True)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_synthetic = a[0,:,0,:,0]
    tau_synthetic = paths.tau.numpy()[0,0,0]
    del paths, a_real, a_imag, a

    # Non-synthetic array
    paths = solver(scene, synthetic_array = False)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_non_synthetic = a[0,:,0,:,0]
    tau_non_synthetic = paths.tau.numpy()[0,:,0,:,0]
    del paths, a_real, a_imag, a

    # Apply phase shift due to different propagation delays between the antenna for
    # the simulation using non-synthetic arrays
    a_non_synthetic = a_non_synthetic*np.exp(-1j*2.*np.pi*(tau_non_synthetic-tau_synthetic)*wavelength)

    max_err = np.max(np.abs(a_synthetic - a_non_synthetic)/np.abs(a_synthetic))
    assert max_err < 1e-2
