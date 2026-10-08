__all__ = ["WaymoLoader"]


def __getattr__(name: str):
	if name == "WaymoLoader":
		from .waymo_loader import WaymoLoader

		return WaymoLoader
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
