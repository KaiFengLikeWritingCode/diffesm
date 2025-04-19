"""
prepare_data.py  --  预处理 NetCDF4 日度气候数据，切分成小块供 DiffESM 训练

主要改动
1. 采用  Dask + chunks={'time': CHUNK_DAYS}  流式读取，避免 OOM
2. save_dataset 直接把每个时间片写入磁盘，减少副本
3. 支持 “只有一个变量(sst)” 的场景；若多变量仍可自动 merge
4. 不再硬编码跳过 r1、r2
5. 默认 NetCDF 压缩（zlib level‑4）
"""

import json
import logging
import os
from typing import List

import dask         # 只要 import 即可触发 xarray Dask 后端
import hydra
import xarray as xr
from omegaconf import DictConfig
from typing_extensions import TypedDict

# ==================== 常量 ==================== #
NUM_CHUNKS = 40          # 输出多少文件
CHUNK_DAYS = 90          # 读取时 time 维度分块大小
ENCODING = lambda ds: {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s"
)

# ==================== 类型 ==================== #
class DATA_DICT(TypedDict):
    load_dir: str
    realizations: dict[str, dict[str, List[str]]]


# ==================== 工具函数 ==================== #
def save_dataset(ds: xr.Dataset, realization: str, save_root: str, num_chunks: int):
    """
    将完整 Dataset 沿 time 维度平均切成 num_chunks 份并分别保存
    """
    outdir = os.path.join(save_root, realization)
    os.makedirs(outdir, exist_ok=True)

    # 清空旧文件
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    n_time = ds.dims["time"]
    step = n_time // num_chunks

    for idx in range(num_chunks):
        start = idx * step
        end = None if idx == num_chunks - 1 else start + step
        sub = ds.isel(time=slice(start, end))

        path = os.path.join(outdir, f"chunk_{idx}.nc")
        logging.info(f"  → writing {path}")
        sub.to_netcdf(path, encoding=ENCODING(sub))


def process_dataset(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """
    · 去除多余 bounds 变量
    · 截取指定年份
    """
    ds = ds.drop_vars(["time_bnds", "lat_bnds", "lon_bnds"], errors="ignore")
    return ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))


def collect_var_data(paths: List[str], base_dir: str) -> xr.Dataset:
    """
    用 open_mfdataset 分块读取同一变量的多个文件
    """
    abs_paths = [os.path.join(base_dir, p) for p in paths]
    ds = xr.open_mfdataset(
        abs_paths,
        combine="by_coords",
        chunks={"time": CHUNK_DAYS},
        parallel=True,
        engine="netcdf4",
    ).sortby("time")

    return ds.drop_vars("time_bnds", errors="ignore")


# ==================== 主程序 ==================== #
@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(cfg: DictConfig):
    json_path = os.path.join(
        cfg.paths.json_data_dir, cfg.esm, cfg.scenario, "data.json"
    )
    with open(json_path) as f:
        data: DATA_DICT = json.load(f)

    load_dir = data["load_dir"]
    save_root = os.path.join(cfg.paths.data_dir, cfg.esm, cfg.scenario)

    logging.info(f"Input  dir : {load_dir}")
    logging.info(f"Output dir : {save_root}")

    for r_name, r_dict in data["realizations"].items():
        logging.info(f"=== Realization {r_name} ===")

        # ---------- 读取并合并变量 ----------
        if len(r_dict) == 1:               # 只有一个变量，例如 'sst'
            var_paths = next(iter(r_dict.values()))
            ds = collect_var_data(var_paths, load_dir)
        else:                              # 多变量：tas + pr …
            dsets = [collect_var_data(p, load_dir) for p in r_dict.values()]
            # 确保 time 对齐
            base_time = dsets[0]["time"]
            for i in range(1, len(dsets)):
                dsets[i] = dsets[i].assign_coords({"time": base_time})
            ds = xr.merge(dsets, compat="override")

        # ---------- 剪裁年份 ----------
        ds = process_dataset(ds, cfg.start_year, cfg.end_year)

        # ---------- 保存 ----------
        save_dataset(ds, r_name, save_root, cfg.num_chunks)

        logging.info(f"Finished realization {r_name}")

    logging.info("✅ All done!")


if __name__ == "__main__":
    main()
