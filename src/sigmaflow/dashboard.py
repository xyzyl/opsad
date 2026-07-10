"""Public entry point for the interactive dashboard (spec §8.3):

    from sigmaflow.dashboard import launch_dashboard
    launch_dashboard(signal=sf, detector="isolation_forest", port=8050)

Requires:  pip install sigmaflow[dashboard]
"""

from .viz.dashboard import create_app, launch_dashboard

__all__ = ["launch_dashboard", "create_app"]
