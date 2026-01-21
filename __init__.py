"""Oceananigans JLD2 读取器（兼容 lesview 结构，自动插值到 x/y/z）。"""

from dataclasses import dataclass
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import xarray as xr


# ----------------------------#
# 数据维度匹配工具
# ----------------------------#


@dataclass
class AxisMatch:
    axis: str  # 'x'/'y'/'z'
    coord: np.ndarray  # 目标坐标（去 halo，且若在界面则已与中心对齐）
    slice_obj: slice  # 用于去 halo 的切片
    source: str  # 'center' 或 'face'
    score: int  # 匹配评分（优先中心，其次去 halo，再是单点）


def _trim(arr: Optional[np.ndarray], halo: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if halo <= 0:
        return arr
    return arr[halo:-halo]


def _load_grid(fdata: h5py.File) -> Dict[str, Dict[str, np.ndarray]]:
    if "grid" not in fdata:
        raise ValueError("文件缺少 grid 信息。")
    g = fdata["grid"]

    def fetch(name: str) -> Optional[np.ndarray]:
        return np.asarray(g[name][()]) if name in g else None

    def halo_val(name: str) -> int:
        if name not in g:
            return 0
        return int(np.asarray(g[name][()]))

    grid = {
        "x": {
            "center_full": fetch("xᶜᵃᵃ"),
            "face_full": fetch("xᶠᵃᵃ"),
            "halo": halo_val("Hx"),
        },
        "y": {
            "center_full": fetch("yᵃᶜᵃ"),
            "face_full": fetch("yᵃᶠᵃ"),
            "halo": halo_val("Hy"),
        },
        "z": {
            "center_full": fetch("zᵃᵃᶜ"),
            "face_full": fetch("zᵃᵃᶠ"),
            "halo": halo_val("Hz"),
        },
    }

    for axis in grid.values():
        axis["center"] = _trim(axis["center_full"], axis["halo"])
        axis["face"] = _trim(axis["face_full"], axis["halo"])
    return grid


def _get_iterations(ts_group: h5py.Group) -> List[str]:
    if "t" in ts_group:
        keys = [k for k in ts_group["t"].keys() if k != "serialized"]
    else:
        keys = []
        for _, grp in ts_group.items():
            if isinstance(grp, h5py.Group):
                keys.extend([k for k in grp.keys() if k != "serialized"])
    ids = []
    for k in keys:
        try:
            ids.append(int(k))
        except (TypeError, ValueError):
            continue
    ids = sorted(set(ids))
    return [str(i) for i in ids]


def _time_coord(ts_group: h5py.Group, iters: Sequence[str], origin: str):
    if "t" not in ts_group:
        return np.arange(len(iters))
    t_group = ts_group["t"]
    seconds = np.full(len(iters), np.nan, dtype=float)
    for i, it in enumerate(iters):
        if it in t_group:
            seconds[i] = t_group[it][()]
    if np.isnan(seconds).all():
        return np.arange(len(iters))
    return pd.to_datetime(seconds, unit="s", origin=origin)


def _slice_for_halo(halo: int) -> slice:
    if halo <= 0:
        return slice(None)
    return slice(halo, -halo)


def _match_axis(size: int, axis: str, grid_axis: Dict[str, np.ndarray]) -> Optional[AxisMatch]:
    center = grid_axis.get("center")
    face = grid_axis.get("face")
    center_full = grid_axis.get("center_full")
    face_full = grid_axis.get("face_full")
    halo = int(grid_axis.get("halo") or 0)

    # 优先直接匹配中心坐标
    if center is not None and size == center.size:
        return AxisMatch(axis, center, slice(None), "center", 4)
    if face is not None and size == face.size:
        return AxisMatch(axis, face, slice(None), "face", 3)
    if center_full is not None and size == center_full.size and halo > 0:
        return AxisMatch(axis, center, _slice_for_halo(halo), "center", 2)
    if face_full is not None and size == face_full.size and halo > 0:
        return AxisMatch(axis, face, _slice_for_halo(halo), "face", 1)
    if size == 1:
        # 单点（slice/surface）情况
        base = center if center is not None else np.array([0.0])
        return AxisMatch(axis, base[:1], slice(None), "center", 0)
    return None


def _candidate_sequences(ndim: int) -> List[Tuple[str, ...]]:
    if ndim == 3:
        return [("z", "y", "x"), ("z", "x", "y"), ("y", "x", "z"), ("x", "y", "z")]
    if ndim == 2:
        return [("z", "y"), ("z", "x"), ("y", "x"), ("x", "y")]
    if ndim == 1:
        return [("z",), ("y",), ("x",)]
    return []


def _infer_matches(shape: Tuple[int, ...], grid: Dict[str, Dict[str, np.ndarray]]) -> List[AxisMatch]:
    if len(shape) == 0:
        return []
    best = None  # (score, matches, priority)
    for priority, seq in enumerate(_candidate_sequences(len(shape))):
        matches = []
        score = 0
        ok = True
        for size, ax in zip(shape, seq):
            m = _match_axis(size, ax, grid.get(ax, {}))
            if m is None:
                ok = False
                break
            matches.append(m)
            score += m.score
        if not ok:
            continue
        if best is None or score > best[0] or (score == best[0] and priority < best[2]):
            best = (score, matches, priority)
    if best is None:
        raise ValueError(f"无法推断变量维度 {shape}")
    return best[1]


def _interp_to_center(data: np.ndarray, source: np.ndarray, target: np.ndarray, axis: int) -> np.ndarray:
    """使用 numpy 沿给定轴线性插值。"""
    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("插值坐标必须是一维。")
    if source.size < 2:
        return np.take(data, indices=[0], axis=axis)

    # 保证升序
    flipped = False
    if source[1] - source[0] < 0:
        source = source[::-1]
        target = target[::-1]
        data = np.flip(data, axis=axis)
        flipped = True

    moved = np.moveaxis(data, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    interp_flat = np.empty((target.size, flat.shape[1]), dtype=data.dtype)
    for i in range(flat.shape[1]):
        interp_flat[:, i] = np.interp(target, source, flat[:, i])
    out = interp_flat.reshape((target.size,) + moved.shape[1:])
    out = np.moveaxis(out, 0, axis)

    if flipped:
        out = np.flip(out, axis=axis)
    return out


# ----------------------------#
# 变量堆叠
# ----------------------------#


def _stack_var(
    name: str,
    group: h5py.Group,
    iters: Sequence[str],
    time_coord,
    grid: Dict[str, Dict[str, np.ndarray]],
) -> Optional[xr.DataArray]:
    first = next((it for it in iters if it in group), None)
    if first is None:
        warnings.warn(f"跳过 {name}: 没有可用迭代。")
        return None

    sample = np.asarray(group[first][()])

    # 纯标量（无空间维度）
    if sample.ndim == 0:
        data = np.full((len(iters),), np.nan, dtype=sample.dtype)
        for ti, it in enumerate(iters):
            if it in group:
                data[ti] = group[it][()]
        return xr.DataArray(data, dims=("time",), coords={"time": time_coord}, name=name)

    matches = _infer_matches(sample.shape, grid)
    slices = tuple(m.slice_obj for m in matches)

    trimmed_shape = sample[slices].shape if slices else sample.shape
    data = np.full(trimmed_shape + (len(iters),), np.nan, dtype=sample.dtype)

    for ti, it in enumerate(iters):
        if it not in group:
            continue
        arr = np.asarray(group[it][()])
        if slices:
            arr = arr[slices]
        if arr.shape != trimmed_shape:
            warnings.warn(f"{name} 的迭代 {it} 形状不匹配，预期 {trimmed_shape}，得到 {arr.shape}")
            continue
        data[..., ti] = arr

    # 将 face 坐标插值到 center
    for axis_idx, m in enumerate(matches):
        if m.source == "face":
            target = grid[m.axis].get("center")
            if target is None:
                warnings.warn(f"{name} 缺少 {m.axis} 中心坐标，保留界面坐标。")
                continue
            data = _interp_to_center(data, m.coord, target, axis_idx)
            matches[axis_idx] = AxisMatch(m.axis, target, slice(None), "center", m.score)

    dims = [m.axis for m in matches] + ["time"]
    coords = {m.axis: m.coord for m in matches}
    coords["time"] = time_coord
    return xr.DataArray(data, dims=dims, coords=coords, name=name)


# ----------------------------#
# 公共 API
# ----------------------------#


def open_dataset(filepath: str, datetime_origin: str = "2000-01-01T00:00:00") -> xr.Dataset:
    """读取 Oceananigans 输出文件（averages / fields / slices 均可）。

    - 自动去 halo；
    - 自动识别变量维度（包含 x/y 可能为 1 的 slice/profile）；
    - 自动将 xi/yi/zi 变量插值到 x/y/z 中心坐标；
    - 返回 xarray.Dataset，变量名与文件一致。
    """
    with h5py.File(filepath, "r") as fdata:
        grid = _load_grid(fdata)
        ts = fdata["timeseries"]
        iters = _get_iterations(ts)
        if not iters:
            raise ValueError("未找到迭代记录（timeseries 组为空）。")
        time_coord = _time_coord(ts, iters, datetime_origin)

        data_vars: Dict[str, xr.DataArray] = {}
        for var_name, var_group in ts.items():
            if var_name in ("t", "serialized"):
                continue
            if not isinstance(var_group, h5py.Group):
                continue
            try:
                da = _stack_var(var_name, var_group, iters, time_coord, grid)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"读取 {var_name} 失败: {exc}")
                continue
            if da is not None:
                data_vars[var_name] = da

        # 仅在“所有变量该维度长度一致且与网格一致”时才附加全局坐标，避免 x/y 为 1 时被强制广播到完整网格
        axis_lengths = {ax: set() for ax in ("x", "y", "z")}
        for da in data_vars.values():
            for ax in ("x", "y", "z"):
                if ax in da.dims:
                    axis_lengths[ax].add(da.sizes[ax])

        coords = {"time": time_coord}
        for ax in ("z", "y", "x"):
            center = grid[ax].get("center")
            if center is None or len(axis_lengths[ax]) == 0:
                continue
            if len(axis_lengths[ax]) == 1 and list(axis_lengths[ax])[0] == center.size:
                coords[ax] = center

        return xr.Dataset(data_vars=data_vars, coords=coords)


def open_oceananigans(filepath: str, datetime_origin: str = "2000-01-01T00:00:00") -> xr.Dataset:
    """向后兼容别名。"""
    return open_dataset(filepath, datetime_origin=datetime_origin)
