from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    tile_id: str
    xyxy: tuple[int, int, int, int]
    image_size: tuple[int, int]


def generate_tiles(width: int, height: int, grid: int = 2, overlap: float = 0.20) -> list[Tile]:
    """Generate overlapping 2x2 or 3x3 tiles in pixel coordinates."""

    if grid < 1:
        raise ValueError("grid must be >= 1")
    overlap = max(0.0, min(0.8, overlap))
    tile_w = int(round(width / grid * (1.0 + overlap)))
    tile_h = int(round(height / grid * (1.0 + overlap)))
    step_x = width / grid
    step_y = height / grid
    tiles = []
    for row in range(grid):
        for col in range(grid):
            cx = int(round((col + 0.5) * step_x))
            cy = int(round((row + 0.5) * step_y))
            x1 = max(0, min(width - 1, cx - tile_w // 2))
            y1 = max(0, min(height - 1, cy - tile_h // 2))
            x2 = min(width, x1 + tile_w)
            y2 = min(height, y1 + tile_h)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            tiles.append(Tile(tile_id=f"r{row}_c{col}", xyxy=(x1, y1, x2, y2), image_size=(width, height)))
    return tiles


def map_tile_box_to_image(
    tile: Tile,
    box_xyxy_norm: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Map a normalized tile prediction back to normalized full-image xyxy."""

    width, height = tile.image_size
    tx1, ty1, tx2, ty2 = tile.xyxy
    tile_w = max(1, tx2 - tx1)
    tile_h = max(1, ty2 - ty1)
    x1, y1, x2, y2 = box_xyxy_norm
    full = (
        (tx1 + x1 * tile_w) / width,
        (ty1 + y1 * tile_h) / height,
        (tx1 + x2 * tile_w) / width,
        (ty1 + y2 * tile_h) / height,
    )
    return tuple(max(0.0, min(1.0, value)) for value in full)
