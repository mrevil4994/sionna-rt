#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Radio map object"""

import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, List
from abc import ABC, abstractmethod

from sionna.rt.utils import watt_to_dbm, log10
from sionna.rt.scene import Scene


class RadioMap(ABC):
    r"""
    Abstract base class for radio maps

    A radio map is generated for the loaded scene for all transmitters using
    a :doc:`radio map solver <radio_map_solvers>`.
    Please refer to the documentation of this module for further details.

    :param scene: Scene for which the radio map is computed
    """

    def __init__(self, scene : Scene):

        self._thermal_noise_power = scene.thermal_noise_power
        self._wavelength = scene.wavelength

        # Positions of the transmitters
        tx_positions_x = mi.Float([tx.position.x[0]
                                for tx in scene.transmitters.values()])
        tx_positions_y = mi.Float([tx.position.y[0]
                                for tx in scene.transmitters.values()])
        tx_positions_z = mi.Float([tx.position.z[0]
                                for tx in scene.transmitters.values()])
        self._tx_positions = mi.Point3f(tx_positions_x,
                                        tx_positions_y,
                                        tx_positions_z)

        # Powers of the transmitters
        self._tx_powers = mi.Float([tx.power[0]
                                    for tx in scene.transmitters.values()])

        # Positions of the receivers
        rx_positions_x = mi.Float([rx.position.x[0]
                                for rx in scene.receivers.values()])
        rx_positions_y = mi.Float([rx.position.y[0]
                                for rx in scene.receivers.values()])
        rx_positions_z = mi.Float([rx.position.z[0]
                                for rx in scene.receivers.values()])
        self._rx_positions = mi.Point3f(rx_positions_x,
                                        rx_positions_y,
                                        rx_positions_z)

        # Sampler used to randomly sample user positions using
        # sample_positions()
        self._sampler = mi.load_dict({'type': 'independent'})

    @property
    @abstractmethod
    def measurement_surface(self):
        r"""Mitsuba rectangle corresponding to the
        radio map measurement plane

        :type: :py:class:`mi.Shape`
        """
        raise NotImplementedError("RadioMap is an abstract class")

    @property
    @abstractmethod
    def cells_count(self):
        r"""Total number of cells in the radio map

        :type: :py:class:`int`
        """
        raise NotImplementedError("RadioMap is an abstract class")

    @property
    @abstractmethod
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system.

        The type of this property depends on the subclass.
        """
        raise NotImplementedError("RadioMap is an abstract class")

    @property
    def num_tx(self):
        r"""Number of transmitters

        :type: :py:class:`int`
        """
        return dr.width(self._tx_positions)

    @property
    def num_rx(self):
        r"""Number of receivers

        :type: :py:class:`int`
        """
        return dr.width(self._rx_positions)

    @property
    @abstractmethod
    def path_gain(self):
        r"""Path gains across the radio map from all transmitters

        The shape of the tensor depends on the subclass.

        :type: :py:class:`mi.TensorXf` with shape `[num_tx, ...]`,
            where the specific dimensions are defined by the subclass.
        """
        raise NotImplementedError("RadioMap is an abstract class")

    @abstractmethod
    def add(
        self,
        e_fields : mi.Vector4f,
        solid_angle : mi.Float,
        array_w : List[mi.Float],
        si_mp : mi.SurfaceInteraction3f,
        k_world : mi.Vector3f,
        tx_indices : mi.UInt,
        hit : mi.Bool
        ) -> None:
        r"""
        Adds the contribution of the rays that hit the measurement plane to the
        radio maps

        The radio maps are updated in place.

        :param e_fields: Electric fields as real-valued vectors of dimension 4
        :param solid_angle: Ray tubes solid angles [sr]
        :param array_w: Weighting used to model the effect of the transmitter
            array
        :param si_mp: Informations about the interaction with the measurement
            plane
        :param k_world: Directions of propagation of the rays
        :param tx_indices: Indices of the transmitters from which the rays
            originate
        :param hit: Flags indicating if the rays hit the measurement plane
        """
        raise NotImplementedError("RadioMap is an abstract class")

    @abstractmethod
    def finalize(self) -> None:
        r"""Finalizes the computation of the radio map"""
        raise NotImplementedError("RadioMap is an abstract class")

    @property
    def rss(self):
        r"""Received signal strength (RSS) across the radio map from all
        transmitters

        The shape of the tensor depends on the subclass.

        :type: :py:class:`mi.TensorXf` with shape `[num_tx, ...]`,
            where the specific dimensions are defined by the subclass.
        """
        n = self.path_gain.ndim
        tx_powers = dr.reshape(mi.TensorXf, self._tx_powers,
                               [self.num_tx] + [1] * (n - 1))
        rss_map = self.path_gain*tx_powers
        return rss_map

    @property
    def sinr(self):
        r"""SINR across the radio map from all transmitters

        The shape of the tensor depends on the subclass.

        :type: :py:class:`mi.TensorXf` with shape `[num_tx, ...]`,
            where the specific dimensions are defined by the subclass.
        """
        rss = self.rss

        # Total received power from all transmitters
        total_pow = dr.sum(rss, axis=0)
        # [1, ...]
        total_pow = dr.reshape(mi.TensorXf, total_pow.array,
                            [1] + list(total_pow.shape))

        # Interference for each transmitter
        # Numerical issue can cause this value to be slightly negative
        interference = total_pow - rss

        # Thermal noise
        noise = self._thermal_noise_power

        # SINR
        sinr_map = rss / (interference + noise)
        return sinr_map

    def tx_association(self, metric : str = "path_gain") -> mi.TensorXi:
        r"""Computes cell-to-transmitter association.

        Each cell is associated with the transmitter providing the highest
        metric, such as path gain, received signal strength (RSS), or
        SINR.

        :param metric: Metric to be used
        :type metric: "path_gain" | "rss" | "sinr"

        :return: Cell-to-transmitter association. The value -1 indicates that
                 there is no coverage for the cell.
        """
        # No transmitter assignment for the cells with no coverage
        tx_association = dr.full(mi.TensorXi, -1, [self.cells_count])


        # Get tensor for desired metric
        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")
        radio_map = getattr(self, metric)

        # Equivalent to argmax
        max_val = dr.tile(dr.max(radio_map, axis=0).array, self.num_tx)
        active = max_val > 0.
        radio_map_flat = radio_map.array
        i = dr.compress((max_val == radio_map_flat) & active)
        if len(i) == 0:
            # No coverage for any cell
            return tx_association

        # Fill the tx association map
        n_tx = mi.Int(i // self.cells_count)
        cell_ind_flat = i % self.cells_count
        dr.scatter(tx_association.array, n_tx, cell_ind_flat)

        return tx_association

    def sample_cells(
        self,
        num_cells : int,
        metric : str = "path_gain",
        min_val_db : float | None = None,
        max_val_db : float | None = None,
        min_dist : float | None = None,
        max_dist : float | None = None,
        tx_association : bool = True,
        seed : int = 1
        ) -> Tuple[mi.TensorXu]:
        # pylint: disable=line-too-long
        r"""Samples random cells in a radio map

        For a given radio map, ``num_cells`` random cells are sampled
        such that the selected metric, e.g., SINR, is
        larger than ``min_val_db`` and/or smaller than ``max_val_db``.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random cells centers to the transmitter under
        consideration.
        By activating the flag ``tx_association``, only cells for which the
        selected metric is the highest across all transmitters are sampled.
        This is useful if one wants to ensure, e.g., that the sampled cells
        for each transmitter provide the highest SINR or RSS.

        :param num_cells: Number of returned random cells for each transmitter

        :param metric: Metric to be considered for sampling cells
        :type metric: "path_gain" | "rss" | "sinr"

        :param min_val_db: Minimum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Only cells for which the selected metric is larger than or equal to
            this value are sampled. Ignored if `None`.

        :param max_val_db: Maximum value for the selected metric ([dB] for path
            gain and SINR; [dBm] for RSS).
            Only cells for which the selected metric is smaller than or equal to
            this value are sampled. Ignored if `None`.

        :param min_dist:  Minimum distance [m] from transmitter for all random
            cells. Ignored if `None`.

        :param max_dist: Maximum distance [m] from transmitter for all random
            cells. Ignored if `None`.

        :param tx_association: If `True`, only cells associated with a
            transmitter are chosen, i.e., cells where the chosen metric is
            the highest among all all transmitters. Else, a user located in a
            sampled cell for a specific transmitter may perceive a higher
            metric from another TX.

        :param seed: Seed for the random number generator

        :return: Cell indices (shape :py:class:`[num_tx, num_cells]`)
            corresponding to the random cells
        """

        num_tx = self.num_tx
        cells_count = self.cells_count

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        if not isinstance(num_cells, int):
            raise ValueError("num_cells must be int.")

        if min_val_db is None:
            min_val_db = float("-inf")
        min_val_db = float(min_val_db)

        if max_val_db is None:
            max_val_db = float("inf")
        max_val_db = float(max_val_db)

        if min_val_db > max_val_db:
            raise ValueError("min_val_d cannot be larger than max_val_db.")

        if min_dist is None:
            min_dist = 0.
        min_dist = float(min_dist)

        if max_dist is None:
            max_dist = float("inf")
        max_dist = float(max_dist)

        if min_dist > max_dist:
            raise ValueError("min_dist cannot be larger than max_dist.")

        # Select metric to be used
        cm = getattr(self, metric)
        cm = dr.reshape(mi.TensorXf, cm, [num_tx, cells_count])

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                cm = 10. * log10(cm)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                cm = watt_to_dbm(cm)

        # Transmitters positions
        tx_pos = self._tx_positions
        tx_pos = dr.ravel([tx_pos.x, tx_pos.y, tx_pos.z])
        # [num_tx, cells_count, 3]
        tx_pos = dr.reshape(mi.TensorXf, tx_pos, [num_tx, 1, 3])

        # Compute distance from each tx to all cells
        # [cells_count, 3]
        cell_centers = self.cell_centers
        # [1, cells_count, 3]
        cell_centers_ = dr.reshape(mi.TensorXf, cell_centers.array,
                                   [1, cells_count, 3])
        # [num_tx, cells_count]
        cell_distance_from_tx = dr.sqrt(dr.sum(dr.square(cell_centers_-tx_pos),
                                               axis=2))

        # [num_tx, cells_count]
        distance_mask = ((cell_distance_from_tx >= min_dist) &
                         (cell_distance_from_tx <= max_dist))

        # Get cells for which metric criterion is valid
        # [num_tx, cells_count]
        cm_mask = (cm >= min_val_db) & (cm <= max_val_db)

        # Get cells for which the tx association is valid
        # [num_tx, cells_count]
        tx_ids = dr.arange(mi.UInt, num_tx)
        tx_ids = dr.reshape(mi.TensorXu, tx_ids, [num_tx, 1])
        tx_a = self.tx_association(metric)
        tx_a = dr.reshape(mi.TensorXu, tx_a, [1, cells_count])
        association_mask = tx_ids == tx_a

        # Compute combined mask
        # [num_tx, cells_count]
        active_cells = distance_mask & cm_mask
        if tx_association:
            active_cells = active_cells & association_mask

        # Loop over transmitters and sample for each transmitters active cells
        self._sampler.seed(seed, num_cells)
        # Sampled positions
        # [num_tx, num_pos, 3]
        sampled_cells = dr.zeros(mi.TensorXu, [num_tx, num_cells])
        scatter_ind = dr.arange(mi.UInt, num_cells)
        for n in range(num_tx):
            active_cells_tx = active_cells[n].array
            # Indices of the active cells for this transmitter
            active_cells_ind = dr.compress(active_cells_tx)
            active_cells_count = dr.width(active_cells_ind)
            if active_cells_count == 0:
                continue
            # Sample cells ids
            # Float in (0,1)
            cell_ids = self._sampler.next_1d()
            # Int
            cell_ids = dr.floor(cell_ids * active_cells_count)
            cell_ids = mi.UInt(cell_ids)
            cell_ids = dr.gather(mi.UInt, active_cells_ind, cell_ids)
            #
            dr.scatter(sampled_cells.array, cell_ids,
                       scatter_ind + n * num_cells)

        return sampled_cells

    def cdf(
        self,
        metric : str = "path_gain",
        tx : int | None = None,
        bins : int = 200
        ) -> Tuple[plt.Figure, mi.TensorXf, mi.Float]:
        r"""Computes and visualizes the CDF of a metric of the radio map

        :param metric: Metric to be shown
        :type metric: "path_gain" | "rss" | "sinr"

        :param tx: Index or name of the transmitter for which to show the radio
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.

        :param bins: Number of bins used to compute the CDF

        :return: Figure showing the CDF

        :return: Data points for the chosen metric

        :return: Cummulative probabilities for the data points
        """

        tensor = self.transmitter_radio_map(metric, tx)
        # Flatten tensor
        tensor = dr.ravel(tensor)

        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                tensor = 10.*log10(tensor)
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                tensor = watt_to_dbm(tensor)

        # Compute the CDF

        # Cells with no coverage are excluded
        active = tensor != float("-inf")
        num_active = dr.count(active)
        # Compute the range
        max_val = dr.max(tensor)
        if max_val == float("inf"):
            raise ValueError("Max value is infinity")
        tensor_ = dr.select(active, tensor, float("inf"))
        min_val = dr.min(tensor_)
        range_val = max_val - min_val
        # Compute the cdf
        ind = mi.UInt(dr.floor((tensor - min_val)*bins/range_val))
        cdf = dr.zeros(mi.UInt, bins)
        dr.scatter_inc(cdf, ind, active)
        cdf = mi.Float(dr.cumsum(cdf))
        cdf /= num_active
        # Values
        x = dr.arange(mi.Float, 1, bins+1)/bins*range_val + min_val

        # Plot the CDF

        fig, _ = plt.subplots()
        plt.plot(x.numpy(), cdf.numpy())
        plt.grid(True, which="both")
        plt.ylabel("Cummulative probability")

        # Set x-label and title
        if metric=="path_gain":
            xlabel = "Path gain [dB]"
            title = "Path gain"
        elif metric=="rss":
            xlabel = "Received signal strength (RSS) [dBm]"
            title = "RSS"
        else:
            xlabel = "Signal-to-interference-plus-noise ratio (SINR) [dB]"
            title = "SINR"
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f' for TX {tx}'

        plt.xlabel(xlabel)
        plt.title(title)

        return fig, x, cdf

    def transmitter_radio_map(
        self,
        metric : str = "path_gain",
        tx : int | None = None
        ) -> mi.TensorXf:
        r"""Returns the radio map values corresponding to transmitter ``tx``
        and a specific ``metric``

        If ``tx`` is `None`, then returns for each cell the maximum value
        accross the transmitters.

        :param metric: Metric for which to return the radio map
        :type metric: "path_gain" | "rss" | "sinr"
        """

        if metric not in ("path_gain", "rss", "sinr"):
            raise ValueError("Invalid metric")
        tensor = getattr(self, metric)

        # Select metric for a specific transmitter or compute max
        if tx is not None:
            if not isinstance(tx, int):
                msg = "Invalid type for `tx`: Must be an int, or None"
                raise ValueError(msg)
            elif (tx >= self.num_tx) or (tx < 0):
                raise ValueError(f"Invalid transmitter index {tx}, expected "
                                 f"index in range [0, {self.num_tx}).")
            tensor = tensor[tx]
        else:
            tensor = dr.max(tensor, axis=0)

        return tensor
