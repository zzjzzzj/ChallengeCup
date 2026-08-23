"""Target detection and incremental-learning utilities for the project."""

# Keep the original four-class taxonomy available for reproducing the r1
# baseline, while exposing the complete r2 class order for continual learning.
# The r2 labels deliberately keep the original ids and append the new classes.
BASE_CLASS_NAMES = ["soldier", "small_aircraft", "warship", "tank"]
INCREMENTAL_CLASS_NAMES = ["patrol_boat", "armored_vehicle"]
ALL_CLASS_NAMES = [*BASE_CLASS_NAMES, *INCREMENTAL_CLASS_NAMES]

# Backwards-compatible alias used by the original r1 preparation/training code.
CLASS_NAMES = BASE_CLASS_NAMES
