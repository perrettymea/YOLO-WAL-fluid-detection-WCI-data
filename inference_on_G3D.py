"""
YOLOv5 inference pipeline for G3D sonar backscatter files (.nc / NetCDF).

For each "ping" (a group inside the NetCDF file), this script:
  1. Normalizes the backscatter data (dB) to an 8-bit grayscale image.
  2. Runs a YOLOv5 detection model on that image (in batches, see below).
  3. Draws detected bounding boxes on a (optionally colormapped) copy of
     the image and saves it to disk.
  4. Converts each detection's pixel coordinates into georeferenced /
     depth-referenced measurements (lat/lon, depth, width, height,
     distance to nadir, etc.) and appends them to a CSV file.

"""

import os
import sys
import time
import json
import shutil
import hashlib
import platform
import datetime
import argparse
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import netCDF4 as nc
from tqdm import tqdm


COLORMAPS = {
    "gray": None,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
}


CSV_HEADER = (
    "lon_mean,lat_mean,h_mean,h_bottom,h_top,layer,ping,width_m,height_m,"
    "distance_to_nadir,reject_box_sidelobe,max_valid_range,distance_image_edge,"
    "xmin,xmax,ymin,ymax,confidence,mean_WC_value_db,std_WC_value_db,"
    "q1_WC_value_db,median_WC_value_db,q3_WC_value_db,percent_90_WC_value_db\n"
)


def db_to_natural(db_values):
    """Convert decibel (dB) values to natural (linear) scale.

    Uses the sonar convention `natural = 10 ** (dB / 20)`.

    Parameters
    ----------
    db_values : array-like or float
        Value(s) expressed in decibels.

    Returns
    -------
    array-like or float
        Value(s) converted to linear scale.
    """
    return 10 ** (db_values / 20)


def natural_to_db(natural_values):
    """Convert natural (linear) scale values back to decibels (dB).

    Values are clipped to a small positive floor (1e-10) before taking the
    log, to avoid `-inf` / NaN results for zero or negative inputs.

    Parameters
    ----------
    natural_values : array-like or float
        Value(s) expressed on a linear scale.

    Returns
    -------
    array-like or float
        Value(s) converted to decibels.
    """
    return 20 * np.log10(np.maximum(natural_values, 1e-10))


def draw_text(img: object, text: str,
              font: object = cv2.FONT_HERSHEY_DUPLEX,
              pos: tuple = (0, 0),
              font_scale: int = 2,
              font_thickness: int = 1,
              text_color: tuple = (255, 255, 255),
              text_color_bg: tuple = (0, 0, 0)
              ) -> object:
    """Draw a text label with a solid background rectangle on an image.

    Typically used to overlay a detection's confidence score near its
    bounding box. The label position is clamped so the rectangle and text
    always stay within the image bounds, even for detections close to an
    edge.

    Parameters
    ----------
    img : np.ndarray
        Image to draw on (modified in place), as used by OpenCV.
    text : str
        Text string to render.
    font : int, optional
        OpenCV font constant. Default cv2.FONT_HERSHEY_DUPLEX.
    pos : tuple, optional
        Desired top-left (x, y) position of the label. May be adjusted to
        stay within the image bounds. Default (0, 0).
    font_scale : int, optional
        Font scale factor used to compute the text size and to render the
        text (the background rectangle is sized to match). Default 2.
    font_thickness : int, optional
        Stroke thickness of the text. Default 1.
    text_color : tuple, optional
        Text color in BGR. Default white.
    text_color_bg : tuple, optional
        Background rectangle color in BGR. Default black.

    Returns
    -------
    tuple
        (text_width, text_height) in pixels, as returned by
        cv2.getTextSize.
    """
    effective_scale = font_scale / 2
    text_size, _ = cv2.getTextSize(text, font, effective_scale, font_thickness)
    text_w, text_h = text_size

    img_h, img_w = img.shape[:2]
    x, y = pos
    x = max(0, min(x, img_w - text_w - 1))
    y = max(0, min(y, img_h - text_h - 7))

    cv2.rectangle(img, (x, y), (x + text_w, y + text_h + 6), text_color_bg, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y + text_h), font, effective_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    return text_size


def creation_folder(name_acquisition: str, FOLDER_RESULT: str) -> None:
    """Create the output folder hierarchy for one inference run.

    Creates, under `FOLDER_RESULT`:
        <name_acquisition>/
            boxes_images/            (annotated PNGs, one subfolder per Layer)
            coord_detections_center/ (per-Layer CSV files with detections)
            processed_markers/       (one empty-ish marker file per fully
                                       processed Layer, used to skip
                                       already-done files on a later run)

    Parameters
    ----------
    name_acquisition : str
        Name/identifier of this inference run; used as the top-level
        output folder name.
    FOLDER_RESULT : str
        Base directory in which to create the output folders.

    Returns
    -------
    None
    """
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition, "boxes_images"), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition, "coord_detections_center"), exist_ok=True)
    os.makedirs(os.path.join(FOLDER_RESULT, name_acquisition, "processed_markers"), exist_ok=True)


def _validate_inputs(args: argparse.Namespace) -> None:
    """Validate CLI/config arguments before any processing starts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (see the `__main__` argument parser).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one or more arguments are invalid. The message lists every
        problem found, not just the first one.
    """
    errors = []

    g3d_path = Path(args.G3D)
    if not g3d_path.is_dir():
        errors.append(f"G3D folder not found or not a directory: {args.G3D}")
    elif not list(g3d_path.glob("*.nc")):
        errors.append(f"No .nc files found in G3D folder: {args.G3D}")

    model_path = Path(args.folder_model) / args.name_model
    if not model_path.is_file():
        errors.append(f"Model weights file not found: {model_path}")

    if args.dB_min >= args.dB_max:
        errors.append(f"dB_min ({args.dB_min}) must be strictly less than dB_max ({args.dB_max})")

    if not (0.0 <= args.confidence_threshold <= 1.0):
        errors.append(f"confidence_threshold must be between 0 and 1, got {args.confidence_threshold}")

    if args.colormap not in COLORMAPS:
        errors.append(f"Unknown colormap '{args.colormap}'. Choose one of: {sorted(COLORMAPS)}")

    if args.size_img <= 0:
        errors.append(f"size_img must be a positive integer, got {args.size_img}")

    if args.batch_size <= 0:
        errors.append(f"batch_size must be a positive integer, got {args.batch_size}")

    if errors:
        raise ValueError("Invalid arguments:\n  - " + "\n  - ".join(errors))


def _resolve_device(device_arg: str) -> str:
    """Resolve the "auto" device choice to an actual torch device string.

    Parameters
    ----------
    device_arg : str
        "auto", "cpu", "cuda", or "cuda:N".

    Returns
    -------
    str
        "cuda:0" if `device_arg` is "auto" and a GPU is available,
        "cpu" if "auto" and no GPU is available, otherwise `device_arg`
        unchanged.
    """
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device_arg


def _print_device_info(device: str) -> None:
    """Print, unambiguously, whether inference will run on CPU or GPU.

    Parameters
    ----------
    device : str
        Resolved device string, e.g. "cpu", "cuda:0".

    Returns
    -------
    None
    """
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"Device: {device} requested, but CUDA is NOT available on this machine -- "
                  f"this will fail. Check your driver/CUDA/torch install.")
            return
        idx = int(device.split(":")[1]) if ":" in device else 0
        name = torch.cuda.get_device_name(idx)
        print(f"Device: GPU -- {device} ({name})")
    else:
        print(f"Device: CPU -- {device} (inference will be noticeably slower than on a GPU)")


def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, read in chunks.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk_size : int, optional
        Bytes read per chunk. Default 1 MiB.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initmodel(model_folder: Path, confidence_threshold: float, name_model: str,
               device: str, force_reload: bool = True):
    """Load a custom YOLOv5 model from a local checkout and move it to a device.

    Parameters
    ----------
    model_folder : Path
        Directory containing the trained weights file.
    confidence_threshold : float
        Minimum confidence score for a detection to be kept (sets
        `model.conf`).
    name_model : str
        Filename of the weights file (e.g. "training_test_with_G3D.pt").
    device : str
        Resolved device string (see `_resolve_device`) the model is moved
        to after loading.
    force_reload : bool, optional
        Whether torch.hub should force a fresh reload of the local
        'yolov5' repo code instead of using its cache. Default True.

    Returns
    -------
    object
        The loaded YOLOv5 model, on `device`, configured for
        class-agnostic NMS and the given confidence threshold.
    """
    print("Model loading")
    model = torch.hub.load(os.path.join(os.path.dirname(__file__), 'yolov5'), 'custom',
                            path=os.path.join(model_folder, name_model), force_reload=force_reload,
                            source='local')
    model = model.to(device)
    model.agnostic = True  # class-agnostic Non-Maximum Suppression
    model.conf = confidence_threshold
    return model


def _compute_detection_record(box_df, i, variable_values, img_shape, lon_port, lon_starboard,
                               lat_port, lat_starboard, immersion, depth, valid_indices,
                               layer, ping):
    """Compute one CSV record (as a string) for a single detection.

    Isolated from `_processfile` so it can be wrapped in a single
    try/except: a degenerate box (e.g. zero pixel height) raises a
    ZeroDivisionError, which the caller can catch and skip without
    losing the rest of the ping's detections.

    Parameters
    ----------
    box_df : pandas.DataFrame
        YOLOv5 detections for the current ping, filtered to class 0.
    i : int
        Row index of the detection to process.
    variable_values : np.ndarray
        Flipped backscatter array (dB) for the current ping.
    img_shape : tuple
        Shape (height, width, channels) of the detection image.
    lon_port, lon_starboard, lat_port, lat_starboard : float
        Longitude/latitude of the port (bâbord) and starboard (tribord)
        edges of the swath, as read from the NetCDF group.
    immersion, depth : float
        Sensor immersion and max depth for the current ping (elevation
        variable), used to convert pixel rows to depth.
    valid_indices : np.ndarray
        Column indices of the first row that are not NaN (used as the
        nadir/reference column).
    layer : str
        Layer name (from the input filename).
    ping : str
        Ping identifier (NetCDF group key).

    Returns
    -------
    str
        A single CSV line (including trailing newline), matching
        `CSV_HEADER`, ready to append to the detections file.

    Raises
    ------
    ZeroDivisionError
        If the detection's pixel height is zero (degenerate box).
    """
    xmin = float(box_df["xmin"][i])
    xmax = float(box_df["xmax"][i])
    ymin = float(box_df["ymin"][i])
    ymax = float(box_df["ymax"][i])
    confidence = float(box_df["confidence"][i])

    lon_mean = (((xmin + xmax) / 2) * (lon_starboard - lon_port)) / img_shape[1] + lon_port
    lat_mean = (((xmin + xmax) / 2) * (lat_starboard - lat_port)) / img_shape[1] + lat_port

    h_top = ((ymin * (depth - immersion)) / img_shape[0]) + immersion
    h_bottom = ((ymax * (depth - immersion)) / img_shape[0]) + immersion

    h_mean = (h_bottom + h_top) / 2
    height_m = np.abs(h_bottom - h_top)

    box_height_px = ymax - ymin
    size_pixel_x = height_m / box_height_px  # raises ZeroDivisionError if box_height_px == 0
    width_m = (xmax - xmin) * size_pixel_x

    box_center_x_px = (xmin + xmax) / 2
    distance_to_nadir = (box_center_x_px - valid_indices[0]) * size_pixel_x

    nadir_column = variable_values[:, valid_indices[0]]
    last_non_nan_index = np.where(~np.isnan(nadir_column))[0][-1]
    max_valid_range = last_non_nan_index / img_shape[0] * (depth - immersion)

    bottom_corners = [(xmin, ymax), (xmax, ymax)]
    reject_box_sidelobe = False
    for x, y in bottom_corners:
        slant = np.sqrt((x - valid_indices[0]) ** 2 + y ** 2)
        if slant > last_non_nan_index:
            reject_box_sidelobe = True
            break

    col_left = int(xmin)
    col_right = int(xmax)
    col_data_left = variable_values[:, col_left]
    col_data_right = variable_values[:, col_right]

    idx_left = np.min(np.where(~np.isnan(col_data_left))[0])
    idx_right = np.min(np.where(~np.isnan(col_data_right))[0])
    h_edge_left = ((idx_left * (depth - immersion)) / img_shape[0]) + immersion
    h_edge_right = ((idx_right * (depth - immersion)) / img_shape[0]) + immersion
    h_edge_min = min(h_edge_left, h_edge_right)
    distance_image_edge = h_edge_min - h_top

    box_WC_value_db = variable_values[int(ymin):int(ymax), int(xmin):int(xmax)].astype(float)
    box_WC_value_natural = db_to_natural(box_WC_value_db)
    box_WC_value_natural_flat = box_WC_value_natural.flatten()

    mean_WC_value_natural = np.nanmean(box_WC_value_natural_flat)
    std_WC_value_natural = np.nanstd(box_WC_value_natural_flat)
    q1_WC_value_natural = np.nanpercentile(box_WC_value_natural_flat, 25)
    median_WC_value_natural = np.nanmedian(box_WC_value_natural_flat)
    q3_WC_value_natural = np.nanpercentile(box_WC_value_natural_flat, 75)
    percent_90_WC_value_natural = np.nanpercentile(box_WC_value_natural_flat, 90)

    mean_WC_value = natural_to_db(mean_WC_value_natural)
    std_WC_value = natural_to_db(std_WC_value_natural)
    q1_WC_value = natural_to_db(q1_WC_value_natural)
    median_WC_value = natural_to_db(median_WC_value_natural)
    q3_WC_value = natural_to_db(q3_WC_value_natural)
    percent_90_WC_value = natural_to_db(percent_90_WC_value_natural)

    return (
        str(lon_mean) + "," + str(lat_mean) + "," + str(h_mean) + "," + str(h_bottom) + "," +
        str(h_top) + "," + layer + "," + "{:05d}".format(int(ping)) + "," +
        str(width_m) + "," + str(height_m) + "," + str(distance_to_nadir) + "," +
        str(reject_box_sidelobe) + "," + str(max_valid_range) + "," +
        str(distance_image_edge) + "," + str(xmin) + "," + str(xmax) + "," + str(ymin) + "," +
        str(ymax) + "," + str(confidence) + "," + str(mean_WC_value) + "," + str(std_WC_value) + "," +
        str(q1_WC_value) + "," + str(median_WC_value) + "," + str(q3_WC_value) + "," +
        str(percent_90_WC_value) + "\n"
    )


def _processfile(model, input_file: Path, FOLDER_RESULT: Path, name_acquisition: str,
                  size_img: int, dB_min: int, dB_max: int, colormap: str = "turbo",
                  batch_size: int = 1, overwrite: bool = False):
    """Run detection on every ping of a single G3D NetCDF file.

    For every ping (NetCDF group) that contains at least one detection of
    class 0, this function saves an annotated PNG and appends one CSV row
    per detection (see the project README / `CSV_HEADER` for the column
    reference). Pings are processed in batches of `batch_size` for faster
    GPU inference.

    Resumability: if a marker for this file's Layer already exists under
    `processed_markers/` and `overwrite` is False, the whole file is
    skipped. Otherwise, any previous partial output for this Layer
    (image folder + CSV) is cleared before (re)processing, and a fresh
    marker is written only once the file has been fully processed
    without error.

    Parameters
    ----------
    model : object
        Loaded YOLOv5 model (see `_initmodel`).
    input_file : Path
        Path to the input file. Only files with a ".nc" extension are
        processed; anything else is silently skipped.
    FOLDER_RESULT : Path
        Base output directory (same one passed to `creation_folder`).
    name_acquisition : str
        Name of this inference run.
    size_img : int
        Image size passed to the YOLOv5 model for inference.
    dB_min : int
        Backscatter value (dB) mapped to pixel intensity 0.
    dB_max : int
        Backscatter value (dB) mapped to pixel intensity 255.
    colormap : str, optional
        Name of an OpenCV colormap (see `COLORMAPS`) applied to the
        *saved* annotated image only. Default "gray".
    batch_size : int, optional
        Number of pings sent to the model in a single inference call.
        Default 1 (one ping at a time, original behavior).
    overwrite : bool, optional
        If True, reprocess this file even if a "done" marker exists.
        Default False.

    Returns
    -------
    dict
        {
          "pings_examined": int,
          "images_saved": int,
          "detections_written": int,
          "skipped": bool,   # True if the file was skipped entirely
        }

    Raises
    ------
    Exceptions related to NetCDF I/O, missing variables/groups, or image
    processing are not caught here. Per-detection measurement errors
    (e.g. degenerate boxes) ARE caught internally and logged instead of
    propagating.
    """
    root, extension = os.path.splitext(input_file.name)
    stats = {"pings_examined": 0, "images_saved": 0, "detections_written": 0, "skipped": False}

    if extension != ".nc":
        return stats

    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choose one of: {sorted(COLORMAPS)}")
    cv2_colormap = COLORMAPS[colormap]

    # Assumes the last 4 characters of the filename (before extension) are
    # a suffix to strip to obtain the "Layer" name used for output
    # subfolders and file names. Adjust here if your naming convention
    # changes.
    layer = root[:-4]

    marker_path = os.path.join(FOLDER_RESULT, name_acquisition, "processed_markers", f"{layer}.done")
    if os.path.exists(marker_path) and not overwrite:
        tqdm.write(f"[skip] {layer}: already processed (marker found) -- use --overwrite to redo it")
        stats["skipped"] = True
        return stats

    tqdm.write(f"process file {input_file} to {FOLDER_RESULT}")

    # (Re)processing this file: clear any previous partial output for
    # this Layer first, so a resumed/forced run never produces duplicate
    # CSV rows or stale images.
    path_to_layer = os.path.join(FOLDER_RESULT, name_acquisition, "boxes_images", layer)
    if os.path.exists(path_to_layer):
        shutil.rmtree(path_to_layer)
    os.makedirs(path_to_layer, exist_ok=True)

    out_file_name = f"{layer}position_detection_with_z_center_and_file_ping.csv"
    out_file_path = os.path.join(FOLDER_RESULT, name_acquisition, "coord_detections_center", out_file_name)
    with open(out_file_path, "w") as f:
        f.write(CSV_HEADER)

    dataset = nc.Dataset(input_file)
    try:
        ping_keys = list(dataset.groups.keys())
        with tqdm(total=len(ping_keys), desc=f"Pings ({layer})", colour="blue",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}", ncols=100, leave=False) as pbar:
            for chunk_start in range(0, len(ping_keys), batch_size):
                chunk_keys = ping_keys[chunk_start: chunk_start + batch_size]
                batch_entries = []

                for ping in chunk_keys:
                    stats["pings_examined"] += 1
                    group = dataset.groups[ping]
                    variable = group.variables['backscatter_mean']
                    raw_values = variable[:]

                    if len(raw_values) == 0:
                        continue
                        
                    if np.ma.isMaskedArray(raw_values):
                        variable_values = raw_values.filled(np.nan).astype(float, copy=True)
                    else:
                        variable_values = np.array(raw_values, dtype=float, copy=True)
 
                    variable_values = np.flipud(variable_values)
                    variable_values_normalized = (variable_values - dB_min) / (dB_max - dB_min) * 255
                    variable_values_normalized = np.clip(variable_values_normalized, 0, 255).astype(np.uint8)
                    variable_values_normalized = np.nan_to_num(variable_values_normalized, nan=0)
                    normalized_array = variable_values_normalized.astype(np.uint8)


                    # Image used for MODEL INFERENCE: always plain
                    # grayscale duplicated across 3 channels, exactly as
                    # the model expects (matches training data format).
                    img = cv2.merge([normalized_array, normalized_array, normalized_array])

                    batch_entries.append({
                        "ping": ping,
                        "group": group,
                        "variable_values": variable_values,
                        "normalized_array": normalized_array,
                        "img": img,
                    })

                pbar.update(len(chunk_keys))

                if not batch_entries:
                    continue

                # Batched inference: one model call for up to
                # `batch_size` pings at once.
                imgs = [entry["img"] for entry in batch_entries]
                results = model(imgs, size=size_img)
                xyxy_per_image = results.pandas().xyxy  # list, same order as `imgs`

                for entry, box_df_full in zip(batch_entries, xyxy_per_image):
                    ping = entry["ping"]
                    group = entry["group"]
                    variable_values = entry["variable_values"]
                    normalized_array = entry["normalized_array"]

                    box_df = box_df_full.loc[np.where(box_df_full["class"] == 0)]
                    box_df = box_df.reset_index(drop=True)
                    if len(box_df) == 0:
                        continue

                    # Image used for DISPLAY/SAVING: colormap applied
                    # here only, never fed back into the model.
                    if cv2_colormap is not None:
                        display_img = cv2.applyColorMap(normalized_array, cv2_colormap)
                    else:
                        display_img = entry["img"].copy()

                    for i in range(0, len(box_df)):
                        p1 = (int(box_df["xmin"][i]), int(box_df["ymin"][i]))
                        p2 = (int(box_df["xmax"][i]), int(box_df["ymax"][i]))
                        cv2.rectangle(display_img, p1, p2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                        draw_text(display_img, str(np.round(float(box_df["confidence"][i]), 2)), font_scale=1,
                                  pos=(int(box_df["xmin"][i]), int(box_df["ymin"][i]) - 22),
                                  text_color_bg=(0, 0, 255))

                    img_color = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                    img2 = Image.fromarray(img_color)

                    # Replace a specific dark-red pixel value with white.
                    # Tuned for the plain grayscale rendering, so only
                    # applied without a colormap (left untouched
                    # otherwise, per your request).
                    if cv2_colormap is None:
                        datas = img2.getdata()
                        newData = []
                        for item in datas:
                            if item[0] == 128 and item[1] == 0 and item[2] == 0:
                                newData.append((255, 255, 255))
                            else:
                                newData.append(item)
                        img2.putdata(newData)

                    file_name = "{}{:05d}.png".format(layer, int(ping))
                    file_path = os.path.join(path_to_layer, file_name)
                    img2.save(file_path)
                    stats["images_saved"] += 1

                    lat_port = group.variables["latitude"][0][0]
                    lat_starboard = group.variables["latitude"][0][1]
                    lon_port = group.variables["longitude"][0][0]
                    lon_starboard = group.variables["longitude"][0][1]
                    immersion = float(group.variables['elevation'][0][0])
                    depth = float(group.variables['elevation'][1][0])

                    valid_indices = np.where(~np.isnan(variable_values[0]))[0]
                    if len(valid_indices) == 0:
                        tqdm.write(f"  [warn] ping {ping} in {layer}: no valid (non-NaN) samples on top row, "
                                   f"skipping geolocation for its detections")
                        continue

                    with open(out_file_path, "a") as myfile:
                        for i in range(0, len(box_df)):
                            try:
                                record = _compute_detection_record(
                                    box_df=box_df, i=i, variable_values=variable_values,
                                    img_shape=img_color.shape, lon_port=float(lon_port),
                                    lon_starboard=float(lon_starboard), lat_port=float(lat_port),
                                    lat_starboard=float(lat_starboard), immersion=immersion, depth=depth,
                                    valid_indices=valid_indices, layer=layer, ping=ping,
                                )
                            except ZeroDivisionError:
                                tqdm.write(f"  [warn] ping {ping} in {layer}, detection {i}: zero-height box "
                                           f"(ymin={box_df['ymin'][i]}, ymax={box_df['ymax'][i]}), skipping")
                                continue
                            except (IndexError, ValueError) as exc:
                                tqdm.write(f"  [warn] ping {ping} in {layer}, detection {i}: could not compute "
                                           f"measurements ({exc!r}), skipping")
                                continue
                            myfile.write(record)
                            stats["detections_written"] += 1

        # Reached only if the whole file was processed without an
        # unhandled exception: mark it as done so a later run can skip it.
        with open(marker_path, "w") as f:
            json.dump({"completed_at": datetime.datetime.now().isoformat(timespec="seconds")}, f)
    finally:
        dataset.close()

    return stats


def _print_run_header(params: dict) -> None:
    """Print an aligned, plain-text summary of the run parameters.

    Deliberately avoids raw ANSI color escape codes: terminals that don't
    interpret them (e.g. classic Windows cmd.exe without virtual terminal
    processing enabled) print the escape sequences as literal garbled
    text instead of colored text. Plain aligned text renders correctly
    everywhere.

    Parameters
    ----------
    params : dict
        Parameter names mapped to their values.

    Returns
    -------
    None
    """
    label_width = max(len(str(k)) for k in params) + 1
    rule = "=" * 60
    print("\n" + rule)
    print("  INFERENCE RUN PARAMETERS")
    print(rule)
    for key, value in params.items():
        print(f"  {str(key).ljust(label_width)}: {value}")
    print(rule + "\n")


def _print_run_summary(nb_files: int, stats: dict, elapsed_seconds: float, manifest_path: str) -> None:
    """Print a final summary once all files have been processed.

    Parameters
    ----------
    nb_files : int
        Number of files that were iterated over.
    stats : dict
        Aggregated stats dict with keys "pings_examined", "images_saved",
        "detections_written", "skipped_files".
    elapsed_seconds : float
        Total wall-clock time spent in `model2data`.
    manifest_path : str
        Path to the run manifest JSON that was written.

    Returns
    -------
    None
    """
    minutes, seconds = divmod(elapsed_seconds, 60)
    rule = "=" * 60
    print("\n" + rule)
    print("  INFERENCE RUN SUMMARY")
    print(rule)
    print(f"  Files processed      : {nb_files - stats['skipped_files']}")
    print(f"  Files skipped (done) : {stats['skipped_files']}")
    print(f"  Pings examined       : {stats['pings_examined']}")
    print(f"  Images saved         : {stats['images_saved']}")
    print(f"  Detections written   : {stats['detections_written']}")
    print(f"  Elapsed time         : {int(minutes)}m {seconds:04.1f}s")
    print(f"  Run manifest         : {manifest_path}")
    print(rule + "\n")


def _build_run_manifest(params: dict, device: str, model_path: Path) -> dict:
    """Build the run manifest dict (before the run) with params/env/model info.

    Parameters
    ----------
    params : dict
        Effective run parameters.
    device : str
        Resolved device string used for inference.
    model_path : Path
        Path to the model weights file.

    Returns
    -------
    dict
        Manifest, to be extended with stats and completed after the run.
    """
    return {
        "timestamp_start": datetime.datetime.now().isoformat(timespec="seconds"),
        "parameters": params,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_used": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "model": {
            "path": str(model_path),
            "sha256": _sha256_of_file(model_path) if model_path.is_file() else None,
        },
    }


def model2data(Folder_model: str, FOLDER_PICTURES: str, FOLDER_RESULT: str, name_model: str,
               name_acquisition: str, size_img: int, confidence_threshold: float,
               dB_min: int, dB_max: int, force_reload: bool = True, colormap: str = "gray",
               batch_size: int = 1, overwrite: bool = False, device: str = "auto") -> int:
    """Run inference on every file in a folder using a YOLOv5 model.

    Loads the model once, then calls `_processfile` on each entry of
    `FOLDER_PICTURES`. Prints a parameter header, an outer per-file
    progress bar, a final run summary, and writes a timestamped run
    manifest JSON to the output folder.

    Parameters
    ----------
    Folder_model : str
        Directory containing the model weights.
    FOLDER_PICTURES : str
        Directory containing the input .nc files.
    FOLDER_RESULT : str
        Directory where results will be written.
    name_model : str
        Weights filename (e.g. "model.pt").
    name_acquisition : str
        Identifier for this inference run.
    size_img : int
        Inference image size passed to the model.
    confidence_threshold : float
        Minimum detection confidence to keep.
    dB_min : int
        dB value mapped to pixel intensity 0.
    dB_max : int
        dB value mapped to pixel intensity 255.
    force_reload : bool, optional
        Passed through to `_initmodel`. Default True.
    colormap : str, optional
        Colormap applied to saved annotated images only. Default "gray".
    batch_size : int, optional
        Number of pings sent to the model per inference call. Default 1.
    overwrite : bool, optional
        Reprocess files even if already marked done. Default False.
    device : str, optional
        "auto", "cpu", "cuda", or "cuda:N". Default "auto".

    Returns
    -------
    int
        Total number of pings examined across all (non-skipped) files.
    """
    if colormap not in COLORMAPS:
        raise ValueError(f"Unknown colormap '{colormap}'. Choose one of: {sorted(COLORMAPS)}")

    resolved_device = _resolve_device(device)
    model_path = Path(Folder_model) / name_model
    model = _initmodel(model_folder=Folder_model, confidence_threshold=confidence_threshold,
                        name_model=name_model, device=resolved_device, force_reload=force_reload)

    run_params = {
        "Folder_model": Folder_model,
        "FOLDER_PICTURES": FOLDER_PICTURES,
        "FOLDER_RESULT": FOLDER_RESULT,
        "name_model": name_model,
        "name_acquisition": name_acquisition,
        "size_img": size_img,
        "confidence_threshold": confidence_threshold,
        "dB_min": dB_min,
        "dB_max": dB_max,
        "colormap": colormap,
        "batch_size": batch_size,
        "overwrite": overwrite,
        "device": device,
    }
    _print_run_header({**run_params, "device (resolved)": resolved_device})
    _print_device_info(resolved_device)
    print("Inference begins")

    manifest = _build_run_manifest(run_params, resolved_device, model_path)
    start_time = time.time()
    total_stats = {"pings_examined": 0, "images_saved": 0, "detections_written": 0, "skipped_files": 0}

    files = sorted(os.listdir(FOLDER_PICTURES))
    outer_pbar = tqdm(files, desc="Files", colour="green",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]", ncols=100)
    for file in outer_pbar:
        outer_pbar.set_postfix_str(file)
        input_file = Path(FOLDER_PICTURES) / Path(file)
        file_stats = _processfile(model=model, input_file=input_file, FOLDER_RESULT=Path(FOLDER_RESULT),
                                   name_acquisition=name_acquisition, size_img=size_img,
                                   dB_min=dB_min, dB_max=dB_max, colormap=colormap,
                                   batch_size=batch_size, overwrite=overwrite)
        total_stats["pings_examined"] += file_stats["pings_examined"]
        total_stats["images_saved"] += file_stats["images_saved"]
        total_stats["detections_written"] += file_stats["detections_written"]
        total_stats["skipped_files"] += int(file_stats["skipped"])

    elapsed_seconds = time.time() - start_time

    manifest["timestamp_end"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(elapsed_seconds, 1)
    manifest["stats"] = total_stats
    manifest_filename = "run_manifest_" + manifest["timestamp_start"].replace(":", "-") + ".json"
    manifest_path = os.path.join(FOLDER_RESULT, name_acquisition, manifest_filename)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _print_run_summary(nb_files=len(files), stats=total_stats, elapsed_seconds=elapsed_seconds,
                        manifest_path=manifest_path)

    return total_stats["pings_examined"]


def _load_config_file(config_path: str) -> dict:
    """Load run parameters from a JSON or YAML file.

    The file's top-level keys must match the CLI argument names (without
    the leading `--`), e.g. `confidence_threshold`, `dB_min`, `colormap`.
    Any unrecognized key is ignored with a warning rather than raising.

    Parameters
    ----------
    config_path : str
        Path to a `.json`, `.yaml`, or `.yml` file.

    Returns
    -------
    dict
        Parsed key/value pairs from the file (possibly empty).

    Raises
    ------
    FileNotFoundError
        If `config_path` does not exist.
    ValueError
        If the file extension is not one of .json/.yaml/.yml.
    ImportError
        If a .yaml/.yml file is given but PyYAML is not installed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_file.suffix.lower()
    if suffix == ".json":
        with open(config_file, "r") as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Reading a .yaml/.yml config file requires PyYAML. "
                "Install it with: pip install pyyaml --break-system-packages"
            ) from exc
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension '{suffix}'. Use .json, .yaml, or .yml")

    return data or {}


if __name__ == "__main__":
    # A lightweight first pass just to detect --config before building the
    # full parser, so values from the config file can be injected as new
    # defaults (CLI flags given explicitly still take priority over them).
    config_pre_parser = argparse.ArgumentParser(add_help=False)
    config_pre_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a .json/.yaml/.yml file with argument values, so you don't have to pass every "
             "flag on the command line. Any flag also given on the command line overrides the file.",
    )
    config_pre_args, _ = config_pre_parser.parse_known_args()

    parser = argparse.ArgumentParser("Inference on G3D files", parents=[config_pre_parser])
    parser.add_argument(
        "--G3D",
        default='G3D',
        type=str,
        help="Where are the G3D you want to infer on. If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--results",
        default="RESULTS",
        type=str,
        help="Where you want to save your results. If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--folder_model",
        default="NETWORKS",
        type=str,
        help="Where are your model weights? If you modify it please put absolute path.",
    )
    parser.add_argument(
        "--name_acquisition",
        default="TEST_INFERENCE",
        type=str,
        help="Name of your inference experiment",
    )
    parser.add_argument(
        "--name_model",
        default="training_test_with_G3D.pt",
        type=str,
        help="Name of the model you use. Please put it with the extension .pt",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.2,
        type=float,
        help="threshold used to discriminate detections made by the network",
    )
    parser.add_argument(
        "--size_img",
        default=960,
        type=int,
        help="Img with be resized with this value before inference. Has to be related to the value used for training",
    )
    parser.add_argument(
        "--dB_min",
        default=-50,
        type=int,
        help="Min dB value to normalize data",
    )
    parser.add_argument(
        "--dB_max",
        default=10,
        type=int,
        help="Max dB value to normalize data",
    )
    parser.add_argument(
        "--colormap",
        default="gray",
        type=str,
        choices=sorted(COLORMAPS),
        help="Colormap applied to the saved annotated PNGs only (detection itself always runs on "
             "plain grayscale). 'gray' reproduces the original rendering.",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Number of pings sent to the model in a single inference call. Higher values are "
             "generally faster on GPU (up to a point limited by VRAM). Default 1 (one ping at a time).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess .nc files even if already marked done from a previous run. Without this "
             "flag, files with an existing marker in processed_markers/ are skipped.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help="Device for inference: 'auto' (default, picks GPU if available else CPU), 'cpu', "
             "'cuda', or 'cuda:N' for a specific GPU index.",
    )
    parser.add_argument(
        "--dump_config",
        default=None,
        type=str,
        help="Write the current effective arguments (defaults + config file + CLI overrides) to "
             "this JSON path and exit, without running inference. Handy to bootstrap a config file.",
    )

    if config_pre_args.config:
        config_values = _load_config_file(config_pre_args.config)
        valid_dests = {action.dest for action in parser._actions}
        unknown_keys = set(config_values) - valid_dests
        if unknown_keys:
            print(f"[warn] Ignoring unknown key(s) in config file: {sorted(unknown_keys)}")
            config_values = {k: v for k, v in config_values.items() if k in valid_dests}
        parser.set_defaults(**config_values)

    args = parser.parse_args()

    if args.dump_config:
        dumped = {k: v for k, v in vars(args).items() if k not in ("config", "dump_config")}
        with open(args.dump_config, "w") as f:
            json.dump(dumped, f, indent=2)
        print(f"Wrote current arguments to {args.dump_config}")
        sys.exit(0)

    try:
        _validate_inputs(args)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")

    FOLDER_PICTURES = args.G3D
    FOLDER_RESULT = args.results
    FOLDER_MODEL = args.folder_model

    creation_folder(args.name_acquisition, FOLDER_RESULT)
    detection = model2data(Folder_model=FOLDER_MODEL, FOLDER_PICTURES=FOLDER_PICTURES,
                            FOLDER_RESULT=FOLDER_RESULT, name_model=args.name_model,
                            size_img=args.size_img, confidence_threshold=args.confidence_threshold,
                            dB_min=args.dB_min, dB_max=args.dB_max,
                            name_acquisition=args.name_acquisition, colormap=args.colormap,
                            batch_size=args.batch_size, overwrite=args.overwrite, device=args.device)
    print("End of inference")
