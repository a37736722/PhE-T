# Taken and modified from https://github.com/Google-Health/genomics-research/blob/main/ml-based-copd/learning/ukb_3066_demo_preprocessing.py

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates an ML-based COPD dataset consisting of a single spirometry blow.

The dataset contains parsed representations of the UKB demo spirometry blow
showcased in field
[3066](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=3066) and a mock COPD
status label. This code demonstrates the how to preprocess these curves for use
with the ML-based COPD model.

This spirometry exhalation volume curve example is publicly available and can
be downloaded following instructions on this page:
https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=3.

"""
from typing import Any

import numpy as np
import pandas as pd

# The relative time scale, in seconds, associated with each series step; 0.01
# denotes that volume was sampled at 10 millisecond intervals.
TIME_SCALE = 0.01

# The volume scale applied to series points; series elements are recorded in ML,
# so a scale of 0.001 converts series elements to L.
VOLUME_SCALE = 0.001

# The number of points in the final ML input representation. Blows shorter than
# `MAX_NUM_POINTS` are right-padded to `MAX_NUM_POINTS` using the last value
# while blows londer than `MAX_NUM_POINTS` are truncated to `MAX_NUM_POINTS`.
MAX_NUM_POINTS = 1000

# The maximum volume value used when interpolating a flow-volume curve of length
# `MAX_NUM_POINTS`. The x-axis on this curve is evenly sampled points from
# `[0, MAX_INTERP_VOLUME]`.
MAX_INTERP_VOLUME = 6.58


def trim_records(records: list[dict[str, Any]]) -> pd.DataFrame:
  """Trims leading zeros from spirometry series to match blow length.

  Each blow is left-padded with a constant number of `num_zero` 0s where
  `len(series) = num_zeros + num_points`. We drop the first `num_zeros-1`
  zeros, keeping the final zero to capture the change in flow from time
  step `t=0` to time step `t=1`.

  Args:
    records: A list of record dictionaries representing individual blows.

  Returns:
    A dataframe containing each trimmed blow.
  """

  for record in records:
    series = record['series']
    num_points = record['num_points']
    num_zeros = len(series) - num_points
    assert {0} == set(series[:num_zeros])
    trimmed_series = series[-(num_points + 1) :]
    assert 0 == trimmed_series[0]
    record['series'] = trimmed_series
  return pd.DataFrame(records)


def compute_volume(series: np.ndarray, volume_scale: float) -> np.ndarray:
  """Rescale `series` to a liter-based volume curve."""
  return (series * volume_scale).astype(np.float32)


def compute_flow(volume: np.ndarray, time_scale: float) -> np.ndarray:
  """Computes flow for the given `volume` array and `time_scale`.

  Flow is the simple first derivative of volume. Note: This should be run before
  right padding to avoid large negative flow values if zero-padded.

  Args:
    volume: The volume-time curve.
    time_scale: The relative time scale, in seconds (i.e., input volume unit per
      second).

  Returns:
    A numpy array representing the corresponding flow curve.
  """
  return np.concatenate(([0.0], np.diff(volume) / time_scale))


def derive_base_curves(
    df: pd.DataFrame,
    volume_scale: float = VOLUME_SCALE,
    time_scale: float = TIME_SCALE,
) -> pd.DataFrame:
  """Derives the base time, volume, and flow curves from a blow series."""
  # Note: We copy the df so that we can rerun this function on the original df.
  df = df.copy()

  # Compute unpadded volume curve, max volume value, and last volume value.
  df['volume'] = df['series'].apply(
      lambda series: compute_volume(  # pylint: disable=g-long-lambda
          series,
          volume_scale,
      )
  )
  df['volume_max'] = df['volume'].apply(np.max)
  df['volume_last'] = df['volume'].apply(lambda volume: volume[-1])

  # Compute unpadded flow curve, max flow value, and last flow value.
  df['flow'] = df['volume'].apply(
      lambda volume: compute_flow(  # pylint: disable=g-long-lambda
          volume,
          time_scale,
      )
  )
  df['flow_max'] = df['flow'].apply(np.max)
  df['flow_last'] = df['flow'].apply(lambda flow: flow[-1])

  df = df.drop(columns=['series'])

  return df


def right_pad_array(
    array: np.ndarray,
    pad_value: float,
    max_num_points: int,
) -> np.ndarray:
  """Right pad the given array with `pad_value` up to `max_num_points`.

  Note: padding is only applied if the array's length is less than
  max_num_points. If the array's length is greater than `max_num_points`, its
  length is truncated to that value.

  Args:
    array: The target array to which padding is applied.
    pad_value: The padding value.
    max_num_points: The target length of the padded array.

  Returns:
    A padded array of length `max_num_points`.
  """
  array = array[: min(len(array), max_num_points)]
  array = np.pad(
      array,
      (0, max_num_points - len(array)),
      mode='constant',
      constant_values=pad_value,
  )
  return array


def compute_time(max_num_points: int, time_scale: float) -> np.ndarray:
  """Retruns a linear array containing `max_num_points` at `time_scale`."""
  return time_scale * np.linspace(
      0,
      max_num_points,
      num=max_num_points,
      endpoint=False,
      dtype=np.float32,
  )


def compute_flow_volume(
    flow: np.ndarray,
    volume: np.ndarray,
    min_volume: float,
    max_volume: float,
    num_points: int,
) -> np.ndarray:
  """Interpolates a flow_volume curve of `num_points`."""
  monotonic_volume = np.maximum.accumulate(volume)
  # Note: this guard ensures that the `xp` argument passed to np.interp is non-
  # increasing. From the documentation: "if the sequence `xp` is non-increasing,
  # interpolation results are meaningless." We relax the strict non-increasing
  # requirement to non-decreasing, as this gives extremely similar results to
  # breaking ties with a small amount of noise (i.e., adding the following to
  # the monotonic curve: `np.linspace(start=1e-4, stop=1e-3, num=num_points)`.
  assert np.all(np.diff(monotonic_volume) >= 0)

  volume_interp_intervals = np.linspace(
      start=min_volume, stop=max_volume, num=num_points
  )
  flow_interp = np.interp(
      volume_interp_intervals, xp=monotonic_volume, fp=flow, left=0, right=0
  )
  return flow_interp


def compute_fef(
    flow: np.ndarray, volume: np.ndarray, volume_max: float
) -> tuple[float, float, float, float]:
  """Computes FEF (forced expiratory flow) values.

  Computes FEF25%, FEF50%, FEF75%, and FEF25-75% values.
  See https://en.wikipedia.org/wiki/Spirometry#Forced_expiratory_flow_(FEF) for
  details.

  Args:
    flow: The flow series.
    volume: The volume series.
    volume_max: The maximum volume (FVC) value.

  Returns:
    A tuple (FEF25%, FEF50%, FEF75%, FEF25-75%).
  """
  flow_size = len(flow)
  assert flow_size == len(volume), 'Flow and Volume lengths do not match.'
  assert flow_size > 1, 'Flow should have more than one values'
  volumes_over_25 = volume >= (0.25 * volume_max)
  volumes_over_50 = volume >= (0.50 * volume_max)
  volumes_over_75 = volume >= (0.75 * volume_max)
  if not any(volumes_over_75):
    raise ValueError(f'Cannot find FEF75 in volume curve: {volume}')

  # Note np.argmax(..) finds the first True value in a boolean array.
  idx_25 = np.argmax(volumes_over_25)
  idx_50 = np.argmax(volumes_over_50)
  idx_75 = np.argmax(volumes_over_75)
  assert 0 <= idx_25 <= idx_50 <= idx_75 < flow_size

  fef25, fef50, fef75 = flow[[idx_25, idx_50, idx_75]]
  fef25_75 = flow[idx_25 : (idx_75 + 1)].mean()
  return fef25, fef50, fef75, fef25_75


def derive_input_representations(
    df: pd.DataFrame,
    max_num_points: int = MAX_NUM_POINTS,
    max_interp_volume: float = MAX_INTERP_VOLUME,
    time_scale: float = TIME_SCALE,
) -> pd.DataFrame:
  """Pads volume and flow to create ML model input representations.

  Note: Padding of both `0` or `row['volume_last']` is applied only when the
  length of the array is less than `max_num_points`. If the array's length is
  greater than `max_num_points`, the array is truncated to length
  `max_num_points`. This guarantees that, if padding is applied, the last value
  in the array is always the last value seen in the first `max_num_points`.

  Args:
    df: The dataframe containing the unpadded volume-time and flow-time curves.
    max_num_points: The length of the ML input representations. Curves larger
      than this value are truncated while curves shorter than this value are
      padded (volume is padded with either `0` or `volume_last`; flow is padded
      with `0`).
    max_interp_volume: The maximum volume value used when interpolating a
      flow-volume curve of length `max_num_points`. The x-axis on this curve is
      evenly sampled points from `[0, max_interp_volume]`.
    time_scale: The scale of each time step in seconds.

  Returns:
    A dataframe consisting of derived input representations.
  """
  # Note: We copy the df so that we can rerun this function on the original df.
  df = df.copy()

  # Compute time. Note: we use an empty apply so that we can get around pandas
  # trying to unpack the `time_curve`.
  df.loc[:, 'time'] = np.NaN
  time_curve = compute_time(max_num_points, time_scale)
  df['time'] = df['time'].apply(lambda _: time_curve)

  # Compute padded volume curves.
  df['volume_pad_last'] = df.apply(
      lambda row: right_pad_array(  # pylint: disable=g-long-lambda
          row['volume'],
          pad_value=row['volume_last'],
          max_num_points=max_num_points,
      ),
      axis=1,
  )

  # Compute padded flow curve.
  df['flow_pad_zero'] = df.apply(
      lambda row: right_pad_array(  # pylint: disable=g-long-lambda
          row['flow'],
          pad_value=0,
          max_num_points=max_num_points,
      ),
      axis=1,
  )

  # Compute padded flow volume curves.
  df['flow_volume_pad_last'] = df.apply(
      lambda row: compute_flow_volume(  # pylint: disable=g-long-lambda
          row['flow_pad_zero'],
          row['volume_pad_last'],
          min_volume=0,
          max_volume=max_interp_volume,
          num_points=max_num_points,
      ),
      axis=1,
  )

  df['blow_fef25'], df['blow_fef50'], df['blow_fef75'], df['blow_fef25_75'] = (
      df.apply(
          lambda row: compute_fef(  # pylint: disable=g-long-lambda
              row['flow'], row['volume'], row['volume_max']
          ),
          axis=1,
          result_type='expand',
      ).T.values
  )

  df = df.drop(columns=['volume', 'flow'])
  return df


def load_spiro_dataframe(ukb_3066_records: list) -> pd.DataFrame:
  """Returns a dataframe containing the preprocessed UKB showcase curve."""
  trimmed_records_df = trim_records(ukb_3066_records)
  base_curve_dfs = derive_base_curves(trimmed_records_df)
  blow_curve_derived_df = derive_input_representations(base_curve_dfs)
  return blow_curve_derived_df