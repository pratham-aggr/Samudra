"""
Generate paper-style validation figures from OM4 ground truth + rollout predictions.zarr.

Intended for PVC paths on Kubernetes (e.g. /data/samudra/...) and local mirrors.
Run: PYTHONPATH=src python -m utils.validation_plots --help
"""

from __future__ import annotations

import argparse
import os
import sys

import cmocean as cm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402
from xarrayutils.plotting import linear_piecewise_scale  # noqa: E402

from utils.notebook import process_data  # noqa: E402


def plot_temp_section(ds_groundtruth, pred_dict, out_path: str) -> None:
    plt.rcParams.update({"font.size": 14})
    new_cmap = cm.cm.thermal
    new_cmap.set_bad("grey", 0.6)
    num_models = len(pred_dict) + 1
    fig, ax = plt.subplots(
        1,
        num_models,
        figsize=(8 * num_models, 3),
        gridspec_kw={"wspace": 0.02, "hspace": 0.5},
    )
    ax = np.atleast_1d(ax)
    vmin, vmax = 0, 30

    da_temp = ds_groundtruth["thetao"]
    section_mask = np.isnan(da_temp).all("x").isel(time=0)
    da_temp_int_x = da_temp.weighted(ds_groundtruth["areacello"]).mean(["x", "time"])
    temp_pred = da_temp_int_x.where(~section_mask)
    temp_pred = temp_pred.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
    temp_pred["y"] = temp_pred.y.assign_attrs(long_name="latitude", units=r"$\degree$")
    temp_pred["lev"] = temp_pred.lev.assign_attrs(long_name="depth", units="m")
    im = temp_pred.plot(ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax[0].invert_yaxis()
    ax[0].set_title("OM4", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[0])
    ax[0].axhline(1000, color="0.5", ls="--")
    ax[0].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

    for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
        da_t = model_data["ds_prediction"]["thetao"]
        sm = np.isnan(da_t).all("x").isel(time=0)
        da_ix = da_t.weighted(ds_groundtruth["areacello"]).mean(["x", "time"])
        tp = da_ix.where(~sm)
        tp = tp.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
        tp["y"] = tp.y.assign_attrs(long_name="latitude", units=r"$\degree$")
        tp["lev"] = tp.lev.assign_attrs(long_name="depth", units="m")
        pred_dict[model_key]["temp_profile"] = tp
        im = tp.plot(ax=ax[j], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[j].invert_yaxis()
        ax[j].set_title(f"{model_data['name']}", fontsize=14)
        linear_piecewise_scale(1000, 5, ax=ax[j])
        ax[j].axhline(1000, color="0.5", ls="--")
        ax[j].set_yticks([])
        ax[j].set_ylabel("")

    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]")
    fig.suptitle("Zonally averaged potential temperature", y=1.02, fontsize=15)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_temp_bias(pred_dict, om4_temp_pred, out_path: str) -> None:
    plt.rcParams.update({"font.size": 14})
    new_cmap = cm.cm.balance
    new_cmap.set_bad("grey", 0.6)
    n = len(pred_dict)
    fig, ax = plt.subplots(1, n, figsize=(8 * n, 3), gridspec_kw={"wspace": 0.02})
    ax = np.atleast_1d(ax)
    vmin, vmax = -0.5, 0.5
    for i, key in enumerate(pred_dict):
        im = (pred_dict[key]["temp_profile"] - om4_temp_pred).plot(
            ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
        )
        ax[i].invert_yaxis()
        ax[i].set_title(f"{pred_dict[key]['name']} − OM4", fontsize=14)
        linear_piecewise_scale(1000, 5, ax=ax[i])
        ax[i].axhline(1000, color="0.5", ls="--")
        if i == 0:
            ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        else:
            ax[i].set_yticks([])
            ax[i].set_ylabel("")
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_salinity_section(ds_groundtruth, pred_dict, out_path: str) -> None:
    plt.rcParams.update({"font.size": 14})
    new_cmap = cm.cm.haline
    new_cmap.set_bad("grey", 0.6)
    num_models = len(pred_dict) + 1
    fig, ax = plt.subplots(
        1,
        num_models,
        figsize=(8 * num_models, 3),
        gridspec_kw={"wspace": 0.02, "hspace": 0.5},
    )
    ax = np.atleast_1d(ax)
    vmin, vmax = 33, 36

    da_s = ds_groundtruth["so"]
    section_mask = np.isnan(da_s).all("x").isel(time=0)
    da_ix = da_s.weighted(ds_groundtruth["areacello"]).mean(["x", "time"])
    salinity_pred = da_ix.where(~section_mask)
    salinity_pred = salinity_pred.rename(r"$S$").assign_attrs(units="psu")
    salinity_pred["y"] = salinity_pred.y.assign_attrs(
        long_name="latitude", units=r"$\degree$"
    )
    salinity_pred["lev"] = salinity_pred.lev.assign_attrs(long_name="depth", units="m")
    om4_salinity_pred = salinity_pred

    im = salinity_pred.plot(
        ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax[0].invert_yaxis()
    ax[0].set_title("OM4", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[0])
    ax[0].axhline(1000, color="0.5", ls="--")
    ax[0].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

    for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
        da_s = model_data["ds_prediction"]["so"]
        sm = np.isnan(da_s).all("x").isel(time=0)
        da_ix = da_s.weighted(ds_groundtruth["areacello"]).mean(["x", "time"])
        sp = da_ix.where(~sm)
        sp = sp.rename(r"$S$").assign_attrs(units="psu")
        sp["y"] = sp.y.assign_attrs(long_name="latitude", units=r"$\degree$")
        sp["lev"] = sp.lev.assign_attrs(long_name="depth", units="m")
        pred_dict[model_key]["salinity_profile"] = sp
        im = sp.plot(ax=ax[j], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax[j].invert_yaxis()
        ax[j].set_title(f"{model_data['name']}", fontsize=14)
        linear_piecewise_scale(1000, 5, ax=ax[j])
        ax[j].axhline(1000, color="0.5", ls="--")
        ax[j].set_yticks([])
        ax[j].set_ylabel("")

    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$S$ [psu]")
    fig.suptitle("Zonally averaged salinity", y=1.02, fontsize=15)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_salinity_bias(pred_dict, om4_salinity_pred, out_path: str) -> None:
    new_cmap = cm.cm.delta
    new_cmap.set_bad("grey", 0.6)
    n = len(pred_dict)
    fig, ax = plt.subplots(1, n, figsize=(8 * n, 3), gridspec_kw={"wspace": 0.02})
    ax = np.atleast_1d(ax)
    vmin, vmax = -0.05, 0.05
    for i, key in enumerate(pred_dict):
        im = (pred_dict[key]["salinity_profile"] - om4_salinity_pred).plot(
            ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
        )
        ax[i].invert_yaxis()
        ax[i].set_title(f"{pred_dict[key]['name']} − OM4", fontsize=14)
        linear_piecewise_scale(1000, 5, ax=ax[i])
        ax[i].axhline(1000, color="0.5", ls="--")
        if i == 0:
            ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        else:
            ax[i].set_yticks([])
            ax[i].set_ylabel("")
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$S$ [psu]")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def ohc_map(ohc_intz):
    return ohc_intz.isel(time=slice(-73, None)).mean("time") - ohc_intz.isel(
        time=slice(0, 73)
    ).mean("time")


def plot_ohc(ds_groundtruth, pred_dict, out_path: str) -> None:
    Days_to_Eq = 0
    c_p = 3850
    rho_0 = 1025
    zeta_joules_factor = 1e21
    plt.rcParams.update({"font.size": 14})
    key = next(iter(pred_dict))
    titles = ["OM4", pred_dict[key]["name"], pred_dict[key]["name"] + " bias"]
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(16, 4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.23},
    )

    def plot_ohc_panel(ax, ohc_data, title, i):
        colormap = cm.cm.balance
        colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
        mean = ohc_data.mean().compute().item()
        std = ohc_data.std().compute().item()
        vmin = mean - 8 * std
        vmax = mean + 8 * std
        im = ax.pcolormesh(
            ohc_data["x"],
            ohc_data["y"],
            ohc_data,
            shading="auto",
            cmap=colormap,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.set_title(title, fontsize=14)
        gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
        if i > 0:
            gl.left_labels = False
        return im

    def plot_diff_ohc(ax, ohc_data, gt_ohc_data, title, i):
        colormap = cm.cm.balance
        colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
        bias_ohc = ohc_data - gt_ohc_data
        mean = ohc_data.mean().compute().item()
        std = ohc_data.std().compute().item()
        vmin = mean - 8 * std
        vmax = mean + 8 * std
        im = ax.pcolormesh(
            bias_ohc["x"],
            bias_ohc["y"],
            bias_ohc,
            shading="auto",
            cmap=colormap,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.set_title(title, fontsize=14)
        gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
        if i > 0:
            gl.left_labels = False
        return im

    datasets = [ds_groundtruth, pred_dict[key]["ds_prediction"]]
    gt_ohc = None
    pred_ohc = None
    for i, (ax, title, ds) in enumerate(zip(axs[:2], titles[:2], datasets)):
        section_mask = np.isnan(ds["thetao"]).all("lev").isel(time=5)
        OHC_pred = (
            (ds["thetao"][Days_to_Eq:] * c_p * rho_0 / zeta_joules_factor)
            .weighted(ds["areacello"] * ds["dz"])
            .sum(["lev"])
            .compute()
        )
        OHC_pred = ohc_map(OHC_pred)
        OHC_pred = OHC_pred.where(~section_mask)
        OHC_pred = OHC_pred.rename("Ocean Heat Content")
        OHC_pred["y"] = OHC_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
        OHC_pred["x"] = OHC_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
        OHC_pred = OHC_pred.assign_attrs(units="ZJ")
        if i == 0:
            gt_ohc = OHC_pred
        else:
            pred_ohc = OHC_pred
        im = plot_ohc_panel(ax, OHC_pred, title, i)

    cbar = fig.colorbar(im, ax=axs[:2], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label("Ocean Heat Content [ZJ]", fontsize=14)

    im = plot_diff_ohc(axs[2], pred_ohc, gt_ohc, titles[2], 2)
    cbar = fig.colorbar(im, ax=axs[2], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label("Ocean Heat Content [ZJ]", fontsize=14)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_sst(ds_groundtruth, pred_dict, out_path: str) -> None:
    plt.rcParams.update({"font.size": 14})
    key = next(iter(pred_dict))
    titles = ["OM4", pred_dict[key]["name"], pred_dict[key]["name"] + " bias"]
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(16, 4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.23},
    )

    def plot_sst_panel(ax, sst_data, title, i):
        colormap = cm.cm.thermal
        colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
        mean = sst_data.mean().compute().item()
        std = sst_data.std().compute().item()
        vmin = mean - std
        vmax = mean + std
        im = ax.pcolormesh(
            sst_data["x"],
            sst_data["y"],
            sst_data,
            shading="auto",
            cmap=colormap,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.set_title(title, fontsize=14)
        gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
        if i > 0:
            gl.left_labels = False
        return im

    def plot_diff_sst(ax, sst_data, gt_sst_data, title, i):
        colormap = cm.cm.balance
        colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
        sst_bias = sst_data - gt_sst_data
        im = ax.pcolormesh(
            sst_bias["x"],
            sst_bias["y"],
            sst_bias,
            shading="auto",
            cmap=colormap,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, edgecolor="black")
        ax.set_title(title, fontsize=14)
        gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}
        gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
        if i > 0:
            gl.left_labels = False
        return im

    datasets = [ds_groundtruth, pred_dict[key]["ds_prediction"]]
    gt_sst = None
    pred_sst = None
    for i, (ax, title, ds) in enumerate(zip(axs[:2], titles[:2], datasets)):
        section_mask = np.isnan(ds["thetao"]).isel(lev=0).isel(time=5)
        SST_pred = ds["thetao"].isel(lev=0).mean("time")
        SST_pred = SST_pred.where(~section_mask)
        SST_pred = SST_pred.rename("2.5m " + r"$\theta_O$")
        SST_pred["y"] = SST_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
        SST_pred["x"] = SST_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
        SST_pred = SST_pred.assign_attrs(units=r"$\degree C$")
        if i == 0:
            gt_sst = SST_pred
        else:
            pred_sst = SST_pred
        im = plot_sst_panel(ax, SST_pred, title, i)

    cbar = fig.colorbar(im, ax=axs[:2], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

    im = plot_diff_sst(axs[2], pred_sst, gt_sst, titles[2], 2)
    cbar = fig.colorbar(im, ax=axs[2], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Save validation PNGs from OM4 + predictions.zarr.")
    p.add_argument(
        "--data-zarr",
        default=os.environ.get("DATA_ZARR", "/data/samudra/data"),
        help="OM4 data zarr path (default: env DATA_ZARR or /data/samudra/data)",
    )
    p.add_argument(
        "--predictions-zarr",
        default=os.environ.get("PREDICTIONS_ZARR", ""),
        help="Rollout predictions.zarr (default: env PREDICTIONS_ZARR)",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Directory for PNG files (default: env OUTPUT_DIR or <rollout>/validation_plots)",
    )
    p.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "Samudra"),
        help="Legend / title name for the model",
    )
    p.add_argument(
        "--time-start",
        default="2014-10-10",
        help="Initial slice on ground-truth time (before alignment with predictions)",
    )
    p.add_argument(
        "--time-end",
        default="2022-12-24",
        help="Initial slice on ground-truth time (before alignment with predictions)",
    )
    args = p.parse_args(argv)

    pred_path = args.predictions_zarr.strip()
    if not pred_path:
        print("ERROR: set --predictions-zarr or PREDICTIONS_ZARR", file=sys.stderr)
        return 1

    out_dir = args.output_dir.strip()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(pred_path)), "validation_plots")
    os.makedirs(out_dir, mode=0o775, exist_ok=True)

    pred_dict = {
        "pred_1": {
            "mode": "thermo_dynamic",
            "name": args.model_name,
            "path": pred_path,
            "ls": ["uo", "vo", "thetao", "so", "zos"],
        },
    }

    train_data = xr.open_dataset(
        args.data_zarr,
        engine="zarr",
        chunks={"time": 10, "lat": 180, "lon": 360},
    )
    train_data = train_data.sel(time=slice(args.time_start, args.time_end))

    ds_groundtruth, pred_dict = process_data(
        train_data,
        pred_dict,
        align_times=True,
        require_rollout_length_600=False,
    )

    print("Rendering figures (dask progress below) …")
    with ProgressBar():
        plot_temp_section(
            ds_groundtruth, pred_dict, os.path.join(out_dir, "fig_temp_section.png")
        )
        section_mask = np.isnan(ds_groundtruth["thetao"]).all("x").isel(time=0)
        da_ix = ds_groundtruth["thetao"].weighted(ds_groundtruth["areacello"]).mean(
            ["x", "time"]
        )
        om4_temp_pred = da_ix.where(~section_mask)
        om4_temp_pred = om4_temp_pred.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
        om4_temp_pred["y"] = om4_temp_pred.y.assign_attrs(
            long_name="latitude", units=r"$\degree$"
        )
        om4_temp_pred["lev"] = om4_temp_pred.lev.assign_attrs(long_name="depth", units="m")

        plot_temp_bias(pred_dict, om4_temp_pred, os.path.join(out_dir, "fig_temp_bias.png"))

        plot_salinity_section(
            ds_groundtruth, pred_dict, os.path.join(out_dir, "fig_salinity_section.png")
        )
        da_s = ds_groundtruth["so"]
        sm = np.isnan(da_s).all("x").isel(time=0)
        da_ix = da_s.weighted(ds_groundtruth["areacello"]).mean(["x", "time"])
        om4_salinity_pred = da_ix.where(~sm)
        om4_salinity_pred = om4_salinity_pred.rename(r"$S$").assign_attrs(units="psu")
        om4_salinity_pred["y"] = om4_salinity_pred.y.assign_attrs(
            long_name="latitude", units=r"$\degree$"
        )
        om4_salinity_pred["lev"] = om4_salinity_pred.lev.assign_attrs(
            long_name="depth", units="m"
        )

        plot_salinity_bias(
            pred_dict, om4_salinity_pred, os.path.join(out_dir, "fig_salinity_bias.png")
        )

        print("Computing OHC / SST maps …")
        plot_ohc(ds_groundtruth, pred_dict, os.path.join(out_dir, "fig_ohc_maps.png"))
        plot_sst(ds_groundtruth, pred_dict, os.path.join(out_dir, "fig_sst_maps.png"))

    print(f"Wrote figures under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
