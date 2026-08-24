"""Render a genuinely dynamic, model-free D7 pipeline video."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from relground.schemas import ObjectObservation


VIDEO_MODE = "dynamic_pipeline"
VIDEO_SEGMENT_RATIOS = (
    ("input_stream", 0.25),
    ("topk_retrieval", 0.20),
    ("sam_masks", 0.25),
    ("observations_3d", 0.30),
)


def image_files(folder: Path) -> list[Path]:
    """Return deterministic image inputs from a scene folder."""
    extensions = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"no images found in: {folder}")
    return paths


def render_stage_video(
    input_paths: Sequence[Path],
    selected_rows: Sequence[dict[str, Any]],
    preview_paths: Sequence[Path],
    observations: Sequence[ObjectObservation],
    artifact_root: Path,
    output_path: Path,
    *,
    fps: float,
    duration_seconds: float,
    codec: str,
) -> dict[str, Any]:
    """Render four continuously animated stages from existing D3-D7 artifacts."""
    if len(preview_paths) != len(selected_rows) or len(preview_paths) < 2:
        raise ValueError("video requires matching selected frames and previews")
    if len(input_paths) < 2:
        raise ValueError("video requires at least two source frames")
    if fps <= 0.0 or not 30.0 <= duration_seconds <= 60.0:
        raise ValueError("video must be 30-60 seconds with positive FPS")
    if len(codec) != 4:
        raise ValueError("OpenCV video codec must contain four characters")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "D7 stage video requires OpenCV; run this command in vggt_geom"
        ) from error

    def load_images(paths: Sequence[Path], label: str) -> list[np.ndarray]:
        loaded: list[np.ndarray] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3:
                raise ValueError(f"cannot decode {label}: {path}")
            loaded.append(image)
        return loaded

    source_images = load_images(input_paths, "source frame")
    preview_images = load_images(preview_paths, "stage preview")
    selected_images: list[np.ndarray] = []
    for row in selected_rows:
        geometry_index = int(row["geometry_index"])
        if not 0 <= geometry_index < len(source_images):
            raise ValueError(
                f"selected geometry index is out of range: {geometry_index}"
            )
        selected_images.append(source_images[geometry_index])

    width, height = 1280, 720
    header_height = 92
    content_height = height - header_height
    palette = np.asarray(
        [
            (76, 204, 255),
            (255, 166, 76),
            (116, 224, 126),
            (218, 122, 255),
        ],
        dtype=np.uint8,
    )

    def smooth(value: float) -> float:
        value = min(1.0, max(0.0, value))
        return value * value * (3.0 - 2.0 * value)

    def fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        canvas = np.full((target_height, target_width, 3), 16, dtype=np.uint8)
        scale = min(
            target_width / image.shape[1],
            target_height / image.shape[0],
        )
        size = (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        )
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interpolation)
        x = (target_width - resized.shape[1]) // 2
        y = (target_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        return canvas

    def zoom(image: np.ndarray, amount: float) -> np.ndarray:
        scale = 1.0 + 0.045 * min(1.0, max(0.0, amount))
        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        y = max(0, (resized.shape[0] - image.shape[0]) // 2)
        x = max(0, (resized.shape[1] - image.shape[1]) // 2)
        return resized[y : y + image.shape[0], x : x + image.shape[1]]

    def blend_sequence(
        images: Sequence[np.ndarray],
        progress: float,
    ) -> tuple[np.ndarray, int]:
        position = min(1.0, max(0.0, progress)) * (len(images) - 1)
        index = min(int(position), len(images) - 2)
        alpha = smooth(position - index)
        first = images[index]
        second = images[index + 1]
        if second.shape[:2] != first.shape[:2]:
            second = cv2.resize(second, (first.shape[1], first.shape[0]))
        blended = cv2.addWeighted(first, 1.0 - alpha, second, alpha, 0.0)
        return blended, index

    def add_header(
        body: np.ndarray,
        title: str,
        subtitle: str,
        overall_progress: float,
    ) -> np.ndarray:
        canvas = np.full((height, width, 3), (20, 23, 30), dtype=np.uint8)
        canvas[header_height:, :] = body
        cv2.putText(
            canvas,
            title,
            (28, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (250, 250, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            subtitle,
            (29, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (178, 190, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (0, 87), (width, 91), (45, 51, 62), -1)
        cv2.rectangle(
            canvas,
            (0, 87),
            (int(round(width * min(1.0, overall_progress))), 91),
            (65, 190, 255),
            -1,
        )
        return canvas

    frame_colors = {
        str(row["frame_id"]): palette[index % len(palette)]
        for index, row in enumerate(selected_rows)
    }
    point_chunks: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    box_corners: list[np.ndarray] = []
    box_colors: list[tuple[int, int, int]] = []
    for observation in observations:
        if not observation.points_ref:
            continue
        point_path = artifact_root / observation.points_ref
        with np.load(point_path, allow_pickle=False) as archive:
            points = np.asarray(archive["points"], dtype=np.float64)
        if len(points) > 1200:
            indices = np.linspace(0, len(points) - 1, 1200, dtype=np.int64)
            points = points[indices]
        color = frame_colors[observation.frame_id]
        point_chunks.append(points)
        point_colors.append(np.repeat(color[None, :], len(points), axis=0))
        box_corners.append(observation.obb.corners())
        box_colors.append(tuple(int(value) for value in color))
    if not point_chunks:
        raise ValueError("video cannot render an empty 3D observation set")
    cloud = np.concatenate(point_chunks, axis=0)
    cloud_colors = np.concatenate(point_colors, axis=0)
    cloud_center = np.median(cloud, axis=0)
    cloud = cloud - cloud_center
    box_corners = [corners - cloud_center for corners in box_corners]
    horizontal_radius = max(
        float(
            np.percentile(
                np.linalg.norm(cloud[:, [0, 2]], axis=1),
                99.5,
            )
        ),
        1e-6,
    )
    vertical_radius = max(
        float(np.percentile(np.abs(cloud[:, 1]), 99.5)),
        1e-6,
    )
    box_edges = [
        (left, right)
        for left in range(8)
        for right in range(left + 1, 8)
        if bin(left ^ right).count("1") == 1
    ]

    total_frames = int(round(duration_seconds * fps))
    segment_frames: list[int] = []
    remaining = total_frames
    for index, (_, ratio) in enumerate(VIDEO_SEGMENT_RATIOS):
        count = (
            remaining
            if index == len(VIDEO_SEGMENT_RATIOS) - 1
            else int(round(total_frames * ratio))
        )
        segment_frames.append(count)
        remaining -= count
    segments = {
        name: count / fps
        for (name, _), count in zip(VIDEO_SEGMENT_RATIOS, segment_frames)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot open video writer for codec {codec}")

    written = 0
    try:
        for (stage_name, _), count in zip(
            VIDEO_SEGMENT_RATIOS,
            segment_frames,
        ):
            for local_index in range(count):
                progress = local_index / max(1, count - 1)
                overall = (written + 1) / total_frames
                if stage_name == "input_stream":
                    image, source_index = blend_sequence(
                        source_images,
                        progress,
                    )
                    body = fit(image, width, content_height)
                    title = "1/4  Input stream"
                    subtitle = (
                        f"{len(source_images)}-frame temporal sweep  |  "
                        f"frame {source_index + 1:02d} -> "
                        f"{min(source_index + 2, len(source_images)):02d}"
                    )
                elif stage_name == "topk_retrieval":
                    slot = min(
                        int(progress * len(selected_images)),
                        len(selected_images) - 1,
                    )
                    slot_progress = progress * len(selected_images) - slot
                    image = zoom(selected_images[slot], slot_progress)
                    left_width = 850
                    body = np.full(
                        (content_height, width, 3),
                        16,
                        dtype=np.uint8,
                    )
                    body[:, :left_width] = fit(
                        image,
                        left_width,
                        content_height,
                    )
                    cv2.putText(
                        body,
                        "PE Top-K ranking",
                        (884, 54),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.67,
                        (245, 245, 245),
                        2,
                        cv2.LINE_AA,
                    )
                    max_score = max(
                        float(row["retrieval_score"])
                        for row in selected_rows
                    )
                    for rank_index, row in enumerate(selected_rows):
                        y = 112 + rank_index * 112
                        active = rank_index == slot
                        color = tuple(
                            int(value)
                            for value in palette[rank_index % len(palette)]
                        )
                        if active:
                            cv2.rectangle(
                                body,
                                (874, y - 38),
                                (1252, y + 51),
                                (43, 49, 61),
                                -1,
                            )
                        cv2.putText(
                            body,
                            f"#{rank_index + 1}  {row['frame_id']}",
                            (894, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.56,
                            color if active else (190, 194, 202),
                            2 if active else 1,
                            cv2.LINE_AA,
                        )
                        score = float(row["retrieval_score"])
                        cv2.rectangle(
                            body,
                            (894, y + 19),
                            (1218, y + 32),
                            (57, 61, 70),
                            -1,
                        )
                        cv2.rectangle(
                            body,
                            (894, y + 19),
                            (
                                894 + int(round(324 * score / max_score)),
                                y + 32,
                            ),
                            color,
                            -1,
                        )
                        cv2.putText(
                            body,
                            f"cos {score:.4f}",
                            (894, y + 51),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (160, 168, 178),
                            1,
                            cv2.LINE_AA,
                        )
                    title = "2/4  Text-to-frame retrieval"
                    subtitle = (
                        f"query: {observations[0].class_text}  |  "
                        f"selected {len(selected_rows)} non-redundant views"
                    )
                elif stage_name == "sam_masks":
                    slot = min(
                        int(progress * len(selected_images)),
                        len(selected_images) - 1,
                    )
                    slot_progress = progress * len(selected_images) - slot
                    original = selected_images[slot]
                    preview = preview_images[slot]
                    if preview.shape[:2] != original.shape[:2]:
                        preview = cv2.resize(
                            preview,
                            (original.shape[1], original.shape[0]),
                        )
                    alpha = 0.08 + 0.92 * smooth(
                        min(1.0, slot_progress * 1.65)
                    )
                    mixed = cv2.addWeighted(
                        original,
                        1.0 - alpha,
                        preview,
                        alpha,
                        0.0,
                    )
                    body = fit(
                        zoom(mixed, slot_progress),
                        width,
                        content_height,
                    )
                    row = selected_rows[slot]
                    frame_id = str(row["frame_id"])
                    count_3d = sum(
                        item.frame_id == frame_id
                        for item in observations
                    )
                    title = "3/4  SAM 3 masks and 3D lifting"
                    subtitle = (
                        f"{frame_id}  |  mask overlay {alpha * 100:4.0f}%  "
                        f"|  {count_3d} valid 3D observations"
                    )
                else:
                    body = np.full(
                        (content_height, width, 3),
                        (10, 13, 19),
                        dtype=np.uint8,
                    )
                    angle = (
                        2.0 * np.pi * progress
                        + np.deg2rad(25.0)
                    )
                    cosine, sine = np.cos(angle), np.sin(angle)
                    rotation_y = np.array(
                        [
                            [cosine, 0.0, sine],
                            [0.0, 1.0, 0.0],
                            [-sine, 0.0, cosine],
                        ]
                    )
                    rotated = cloud @ rotation_y.T
                    view_width = 1020
                    view_height = content_height - 42
                    scale = min(
                        view_width * 0.43 / horizontal_radius,
                        view_height * 0.43 / vertical_radius,
                    )
                    px = np.rint(
                        view_width / 2 + rotated[:, 0] * scale
                    ).astype(np.int32)
                    py = (
                        np.rint(
                            view_height / 2 - rotated[:, 1] * scale
                        ).astype(np.int32)
                        + 18
                    )
                    valid = (
                        (px >= 2)
                        & (px < view_width - 2)
                        & (py >= 2)
                        & (py < content_height - 2)
                    )
                    order = np.argsort(rotated[:, 2])
                    order = order[valid[order]]
                    for dx, dy in (
                        (0, 0),
                        (1, 0),
                        (0, 1),
                        (1, 1),
                    ):
                        body[
                            py[order] + dy,
                            px[order] + dx,
                        ] = cloud_colors[order]
                    for corners, color in zip(box_corners, box_colors):
                        rotated_corners = corners @ rotation_y.T
                        corner_x = np.rint(
                            view_width / 2
                            + rotated_corners[:, 0] * scale
                        ).astype(int)
                        corner_y = (
                            np.rint(
                                view_height / 2
                                - rotated_corners[:, 1] * scale
                            ).astype(int)
                            + 18
                        )
                        for left, right in box_edges:
                            cv2.line(
                                body,
                                (
                                    int(corner_x[left]),
                                    int(corner_y[left]),
                                ),
                                (
                                    int(corner_x[right]),
                                    int(corner_y[right]),
                                ),
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                    cv2.putText(
                        body,
                        "FRAME COLORS",
                        (1054, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.47,
                        (205, 210, 220),
                        1,
                        cv2.LINE_AA,
                    )
                    for rank_index, row in enumerate(selected_rows):
                        y = 110 + rank_index * 66
                        color = tuple(
                            int(value)
                            for value in palette[rank_index % len(palette)]
                        )
                        cv2.circle(
                            body,
                            (1070, y),
                            8,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            body,
                            str(row["frame_id"]),
                            (1090, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.43,
                            (190, 198, 210),
                            1,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        body,
                        "colored points = lifted pixels",
                        (1038, 431),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (150, 160, 174),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        body,
                        "wireframes = 3D OBBs",
                        (1038, 458),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (150, 160, 174),
                        1,
                        cv2.LINE_AA,
                    )
                    title = "4/4  Cached 3D ObjectObservations"
                    subtitle = (
                        f"{len(observations)} observations  |  "
                        f"{sum(len(chunk) for chunk in point_chunks):,} "
                        "sampled points  |  rotating world view"
                    )
                writer.write(
                    add_header(body, title, subtitle, overall)
                )
                written += 1
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("stage video was not created")
    return {
        "mode": VIDEO_MODE,
        "duration_seconds": written / fps,
        "fps": fps,
        "codec": codec,
        "segments": segments,
        "source_frame_count": len(source_images),
        "selected_frame_count": len(selected_images),
        "observation_count": len(observations),
        "rendered_point_count": int(
            sum(len(chunk) for chunk in point_chunks)
        ),
    }
