"""Simplified 2D cross-section of an automotive engine bay.

Models a top-down view of a typical engine compartment with major components
represented as geometric obstacles. All dimensions in decimeters (dm) for
convenient visualization in a ~8x6 workspace.

Layout (approximate):
    +---------------------------------------------------+
    |  Firewall (top boundary)                          |
    |                                                   |
    |   [Battery]         [Intake Manifold]             |
    |                                                   |
    |          [=====Engine Block=====]                 |
    |                                                   |
    |   [ABS]     [Exhaust]    [Alternator]            |
    |                                                   |
    |  Bumper (bottom boundary)                         |
    +---------------------------------------------------+

Routing task: ECU connector (rear-left) -> Sensor (front-right)
"""

import numpy as np


def engine_bay():
    """Create engine bay 2D cross-section scenario.

    Returns:
        (obstacles, bounds, resolution, start, goal, waypoints, metadata)
    """
    obstacles = [
        # Engine block - large central rectangle
        {"type": "rectangle", "center": [4.0, 3.0], "half_extents": [1.8, 1.0],
         "label": "Engine Block"},

        # Battery box - upper left
        {"type": "rectangle", "center": [1.2, 4.5], "half_extents": [0.7, 0.5],
         "label": "Battery"},

        # Intake manifold - upper right
        {"type": "rectangle", "center": [6.0, 4.5], "half_extents": [1.0, 0.2],
         "label": "Intake Manifold"},

        # Exhaust pipe - front center (circular cross-section)
        {"type": "circle", "center": [3.8, 1.2], "radius": 0.5,
         "label": "Exhaust Pipe"},

        # ABS module - front left
        {"type": "rectangle", "center": [1.3, 1.5], "half_extents": [0.5, 0.5],
         "label": "ABS Module"},

        # Alternator - front right
        {"type": "circle", "center": [6.2, 1.5], "radius": 0.55,
         "label": "Alternator"},

        # Coolant reservoir - mid left
        {"type": "circle", "center": [0.8, 3.0], "radius": 0.35,
         "label": "Coolant Reservoir"},

        # Power steering pump - mid right
        {"type": "circle", "center": [7.0, 3.0], "radius": 0.4,
         "label": "PS Pump"},

        # Fuse box - upper left near firewall
        {"type": "rectangle", "center": [2.5, 5.2], "half_extents": [0.4, 0.25],
         "label": "Fuse Box"},
    ]

    bounds = ((0, 8), (0, 6))
    resolution = 0.03

    # ECU connector: near firewall, left side
    start = np.array([0.5, 5.5])

    # Temperature sensor: front right, near alternator
    goal = np.array([7.5, 0.5])

    # Preferred waypoints (clip/clamp attachment points along the route)
    waypoints = [
        np.array([1.5, 5.0]),   # Clip near fuse box
        np.array([2.0, 3.0]),   # Clip on left side of engine
        np.array([5.5, 1.0]),   # Clip near exhaust routing
    ]

    metadata = {
        "description": "ECU to Temperature Sensor routing",
        "cable_spec": "AWG16 automotive, 4x bend radius",
        "obstacle_labels": {i: obs.get("label", f"Obstacle {i}")
                            for i, obs in enumerate(obstacles)},
    }

    return obstacles, bounds, resolution, start, goal, waypoints, metadata


def engine_bay_multi_route():
    """Engine bay scenario with multiple routing tasks.

    Three cable routes:
    1. ECU -> Temperature sensor (rear-left to front-right)
    2. Battery -> Starter motor (upper-left to lower-center)
    3. Fuse box -> Headlight (upper-center to front-left)
    """
    base = engine_bay()
    obstacles, bounds, resolution = base[0], base[1], base[2]

    routes = [
        {
            "name": "ECU -> Temp Sensor",
            "start": np.array([0.5, 5.5]),
            "goal": np.array([7.5, 0.5]),
            "color": "lime",
            "awg": 18,
        },
        {
            "name": "Battery -> Starter",
            "start": np.array([1.2, 5.2]),
            "goal": np.array([3.5, 0.5]),
            "color": "cyan",
            "awg": 10,
        },
        {
            "name": "Fuse Box -> Headlight",
            "start": np.array([2.5, 5.5]),
            "goal": np.array([0.5, 0.5]),
            "color": "yellow",
            "awg": 16,
        },
    ]

    return obstacles, bounds, resolution, routes
