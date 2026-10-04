from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch


TOKEN_COLORS = {
    "I": "tab:blue",
    "love": "tab:orange",
    "robots": "tab:green",
}


def save_figure(
    figure,
    output_path,
    dpi=180,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.35,
    )

    plt.close(
        figure
    )


def set_equal_axes(
    axis,
    points,
    margin_ratio=0.18,
):
    points = points.detach().cpu()

    mins = points.min(
        dim=0
    ).values

    maxs = points.max(
        dim=0
    ).values

    center = (
        mins + maxs
    ) / 2.0

    span = (
        maxs - mins
    ).max().item()

    if span < 0.5:
        span = 0.5

    radius = (
        span / 2.0
        * (1.0 + 2.0 * margin_ratio)
    )

    axis.set_xlim(
        center[0].item() - radius,
        center[0].item() + radius,
    )

    axis.set_ylim(
        center[1].item() - radius,
        center[1].item() + radius,
    )

    axis.set_zlim(
        center[2].item() - radius,
        center[2].item() + radius,
    )

    axis.set_box_aspect(
        (1, 1, 1)
    )


def style_3d_axis(
    axis,
    title,
):
    axis.set_title(
        title,
        pad=18,
    )

    axis.set_xlabel(
        "dim 0",
        labelpad=8,
    )

    axis.set_ylabel(
        "dim 1",
        labelpad=8,
    )

    axis.set_zlabel(
        "dim 2",
        labelpad=8,
    )

    axis.scatter(
        0.0,
        0.0,
        0.0,
        color="black",
        s=28,
        zorder=5,
    )

    axis.text(
        0.0,
        0.0,
        0.0,
        "  O",
        color="black",
    )

    axis.view_init(
        elev=24,
        azim=-58,
    )


def draw_vector(
    axis,
    start,
    end,
    color,
    label=None,
    linewidth=2.6,
    label_offset=(0.04, 0.04, 0.04),
    label_box=True,
):
    start = start.detach().cpu()
    end = end.detach().cpu()

    vector = (
        end - start
    )

    axis.quiver(
        start[0].item(),
        start[1].item(),
        start[2].item(),
        vector[0].item(),
        vector[1].item(),
        vector[2].item(),
        color=color,
        arrow_length_ratio=0.08,
        linewidth=linewidth,
    )

    axis.scatter(
        end[0].item(),
        end[1].item(),
        end[2].item(),
        color=color,
        s=42,
        zorder=6,
    )

    if label is not None:
        dx = label_offset[0]
        dy = label_offset[1]
        dz = label_offset[2]

        bbox = None

        if label_box:
            bbox = {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.2,
            }

        axis.text(
            end[0].item() + dx,
            end[1].item() + dy,
            end[2].item() + dz,
            label,
            color=color,
            fontsize=10,
            bbox=bbox,
        )


def pca_project_to_3d(
    vectors,
    center=True,
):
    vectors = vectors.detach().cpu()

    working_vectors = vectors

    if center:
        mean = vectors.mean(
            dim=0,
            keepdim=True,
        )

        working_vectors = (
            vectors - mean
        )

    if working_vectors.shape[1] <= 3:
        if working_vectors.shape[1] == 3:
            return working_vectors

        padding = torch.zeros(
            working_vectors.shape[0],
            3 - working_vectors.shape[1],
        )

        return torch.cat(
            [
                working_vectors,
                padding,
            ],
            dim=1,
        )

    U, S, Vh = torch.linalg.svd(
        working_vectors,
        full_matrices=False,
    )

    components = Vh[
        :3
    ].T

    projected = (
        working_vectors
        @ components
    )

    return projected



def configure_korean_font():
    candidate_names = [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "NanumBarunGothic",
        "Malgun Gothic",
        "AppleGothic",
    ]

    font_paths = font_manager.findSystemFonts(
        fontext="ttf"
    )

    otf_paths = font_manager.findSystemFonts(
        fontext="otf"
    )

    for font_path in otf_paths:
        font_paths.append(
            font_path
        )

    for font_path in font_paths:
        try:
            font_name = (
                font_manager
                .FontProperties(
                    fname=font_path
                )
                .get_name()
            )
        except Exception:
            continue

        for candidate_name in candidate_names:
            if candidate_name in font_name:
                plt.rcParams[
                    "font.family"
                ] = font_name

                plt.rcParams[
                    "axes.unicode_minus"
                ] = False

                print(
                    "Matplotlib Korean font:",
                    font_name,
                )

                return font_name

    print(
        "Korean font was not found. "
        "Korean labels may appear as boxes."
    )

    return None
