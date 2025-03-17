#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import mitsuba as mi
import numpy as np
import drjit as dr

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
from sionna.rt.utils import r_hat, subcarrier_frequencies
from scipy.constants import c as SPEED_OF_LIGHT


####################################################
# Utilities
####################################################

def compute_doppler_spectrum(scene, paths):
    num_subcarriers = 128
    subcarrier_spacing = 15e3
    bandwidth = num_subcarriers * subcarrier_spacing
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    num_time_steps = 1000
    sampling_frequency = bandwidth / num_subcarriers
    sampling_time = 1/sampling_frequency

    h_freq = paths.cfr(frequencies=frequencies,
                       sampling_frequency=sampling_frequency,
                       num_time_steps=num_time_steps,
                       normalize=True,
                       out_type="numpy")

    # Compute Doppler spectrum
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_bins, num_subcarriers]
    ds = np.square(np.abs(np.fft.fft(h_freq, axis=-2)))
    ds = np.fft.fftshift(ds, axes=[-2])
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_bins]
    ds = np.mean(ds, axis=-1)

    # Compute delay-to-speed transformation
    doppler_resolution = 1/(sampling_time*num_time_steps)
    if num_time_steps % 2 == 0:
        start=-num_time_steps/2
        limit=num_time_steps/2
    else:
        start=-(num_time_steps-1)/2
        limit=(num_time_steps-1)/2+1

    doppler_frequencies = np.arange(start=start, stop=limit) * doppler_resolution
    velocities = doppler_frequencies/scene.frequency.numpy()[0]*SPEED_OF_LIGHT

    return velocities, ds


####################################################
# Unit tests
####################################################

def test_moving_tx():
    """Test that the TX speed can be correctly estimated from the Doppler spectrum"""

    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="V")
    scene.rx_array = scene.tx_array

    v = 100
    scene.add(Transmitter(name="tx",
                        position=[0,0,0],
                        orientation=[0,0,0],
                        velocity=[v,0,0]))
    scene.add(Receiver(name="rx",
                    position=[10,0,0],
                    orientation=[0,0,0],
                    velocity=[0,0,0]))

    solver = PathSolver()
    paths = solver(scene)

    velocities, ds = compute_doppler_spectrum(scene, paths)
    ds = np.squeeze(ds)
    v_hat = velocities[np.argmax(ds)]

    v_ref = v
    assert np.abs(v_hat-v_ref)/np.abs(v_ref) <= 0.05

def test_moving_rx():
    """Test that the RX speed can be correctly estimated from the Doppler spectrum"""

    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")
    scene.rx_array = scene.tx_array

    v = -100
    scene.add(Transmitter(name="tx",
                        position=[0,0,0],
                        orientation=[0,0,0],
                        velocity=[0,0,0]))
    scene.add(Receiver(name="rx",
                        position=[10,0,0],
                        orientation=[0,0,0],
                        velocity=[v,0,0]))

    solver = PathSolver()
    paths = solver(scene)

    velocities, ds = compute_doppler_spectrum(scene, paths)
    ds = np.squeeze(ds)
    v_hat = velocities[np.argmax(ds)]

    v_ref = -v
    assert np.abs(v_hat-v_ref)/np.abs(v_ref) <= 0.05

def test_moving_tx_rx():
    """Test that the differentia TX-RX speed can be correctly estimated from the Doppler spectrum"""

    scene = load_scene()
    scene.tx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")
    scene.rx_array = scene.tx_array

    v_rx = 10
    v_tx = 30
    scene.add(Transmitter(name="tx",
                          position=[0,0,0],
                          orientation=[0,0,0],
                          velocity=[v_tx,0,0]))
    scene.add(Receiver(name="rx",
                       position=[10,0,0],
                       orientation=[0,0,0],
                       velocity=[v_rx,0,0]))

    solver = PathSolver()
    paths = solver(scene)

    velocities, ds = compute_doppler_spectrum(scene, paths)
    ds = np.squeeze(ds)
    v_hat = velocities[np.argmax(ds)]

    v_ref = v_tx - v_rx
    assert np.abs(v_hat-v_ref)/np.abs(v_ref) <= 0.05

def test_multi_tx_rx_synthetic_array():
    """Check that the doppler spectra for all pairs of antennas of each link
        match, in a multi-TX multi-RX scenario with a synthetic array.
    """

    scene = load_scene(sionna.rt.scene.simple_wedge)

    scene.tx_array = PlanarArray(num_rows=2,
                            num_cols=2,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")
    scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=2,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")

    tx_velocities = np.zeros([2, 3])
    tx_velocities[0] = [0, 0, 0]
    tx_velocities[1] = [110, 0, 0]
    scene.add(Transmitter(name="tx1",
                position=[20,-20,0],
                orientation=[0,0,0],
                velocity=tx_velocities[0]))
    scene.add(Transmitter(name="tx2",
                position=[10,-30,0],
                orientation=[0,0,0],
                velocity=tx_velocities[0]))

    rx_velocities = np.zeros([3,3])
    rx_velocities[0] = [10, 10, 0]
    rx_velocities[1] = [-50, -50, 0]
    scene.add(Receiver(name="rx1",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[0]))
    scene.add(Receiver(name="rx2",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[1]))
    scene.add(Receiver(name="rx3",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[2]))

    solver = PathSolver()
    paths = solver(scene, max_depth=1, los=False, synthetic_array=True)

    velocities, ds = compute_doppler_spectrum(scene, paths)
    # [num_rx, 1, num_tx, 1, 1]
    ds_max = np.max(np.abs(ds), axis=(1, 3, 4), keepdims=True)
    ds = np.where(ds<ds_max/0.5, 0.0, 1.0)

    for i, _ in enumerate(scene.receivers):
        for j, _ in enumerate(scene.transmitters):
            ref = ds[i,0,j,0]
            for k in range(scene.rx_array.num_ant):
                for l in range(scene.tx_array.num_ant):
                    assert(np.all(ref == ds[i,k,j,l]))

def test_multi_tx_rx_non_synthetic_array():
    """Check that the doppler spectra for all pairs of antennas of each link
        match, in a multi-TX multi-RX scenario with a non-synthetic array.
    """

    scene = load_scene(sionna.rt.scene.simple_wedge)

    scene.tx_array = PlanarArray(num_rows=2,
                            num_cols=2,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")
    scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=2,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")

    tx_velocities = np.zeros([2, 3])
    tx_velocities[0] = [0, 0, 0]
    tx_velocities[1] = [110, 0, 0]
    scene.add(Transmitter(name="tx1",
                position=[20,-20,0],
                orientation=[0,0,0],
                velocity=tx_velocities[0]))
    scene.add(Transmitter(name="tx2",
                position=[10,-30,0],
                orientation=[0,0,0],
                velocity=tx_velocities[0]))

    rx_velocities = np.zeros([3,3])
    rx_velocities[0] = [10, 10, 0]
    rx_velocities[1] = [-50, -50, 0]
    scene.add(Receiver(name="rx1",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[0]))
    scene.add(Receiver(name="rx2",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[1]))
    scene.add(Receiver(name="rx3",
            position=[20,-20, 0],
            orientation=[0,0,0],
            velocity=rx_velocities[2]))

    solver = PathSolver()
    paths = solver(scene, max_depth=1, los=False, synthetic_array=False)

    velocities, ds = compute_doppler_spectrum(scene, paths)
    ds_max = np.max(np.abs(ds), axis=(1, 3, 4), keepdims=True)
    ds = np.where(ds<ds_max/0.5, 0.0, 1.0)

    for i, _ in enumerate(scene.receivers):
        for j, _ in enumerate(scene.transmitters):
            ref = ds[i,0,j,0]
            for k in range(scene.rx_array.num_ant):
                for l in range(scene.tx_array.num_ant):
                    assert(np.all(ref==ds[i,k,j,l]))

def test_moving_reflector():
    """Test that moving reflector has the right Doppler shift"""

    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)

    scene.get("reflector").velocity = [0, 0, -20]

    scene.tx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")
    scene.rx_array = scene.tx_array

    scene.add(Transmitter("tx", [-25,0.1,50]))
    scene.add(Receiver("rx",    [ 25,0.1,50]))

    # Compute the reflected path
    solver = PathSolver()
    paths = solver(scene, max_depth=1, los=False)

    # Compute theoretical Doppler shift for this path
    theta_t = np.squeeze(paths.theta_t.numpy())
    phi_t = np.squeeze(paths.phi_t.numpy())
    k_0 = r_hat(theta_t, phi_t)

    theta_r = np.squeeze(paths.theta_r.numpy())
    phi_r = np.squeeze(paths.phi_r.numpy())
    k_1 = -r_hat(theta_r, phi_r)

    doppler_theo = np.sum((k_1-k_0)*scene.get("reflector").velocity)/scene.wavelength
    assert np.isclose(np.squeeze(paths.doppler.numpy()), doppler_theo)
