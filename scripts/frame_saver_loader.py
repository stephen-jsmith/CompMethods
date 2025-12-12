from __future__ import annotations

from pathlib import Path
import sys
import pickle
from typing import Any, Union

# Ensure project root is on sys.path so `objects` package imports work
# when running this script directly from other working directories.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def save_frame(
    frame: Any, path: Union[str, Path], protocol: int = pickle.HIGHEST_PROTOCOL
) -> None:
    """Serialize and save a frame element to disk using pickle.

    Parameters
    - frame: Any Python object representing a frame element (e.g., `Member2D`).
    - path: Destination file path where the object will be written.
    - protocol: Pickle protocol to use (defaults to highest available).

    The function will create parent directories if they do not exist.
    """
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("wb") as fh:
        pickle.dump(frame, fh, protocol=protocol)


def load_frame(path: Union[str, Path]) -> Any:
    """Load and return a previously saved frame element from disk.

    Raises `FileNotFoundError` when the file does not exist and `pickle.UnpicklingError`
    when the file is not a valid pickle for an object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    with p.open("rb") as fh:
        obj = pickle.load(fh)

    return obj


__all__ = ["save_frame", "load_frame"]


if __name__ == "__main__":
    # Minimal CLI to create an example frame and save it, or to load an existing file.
    import argparse

    parser = argparse.ArgumentParser(
        description="Save/load a frame element (Member/Element) via pickle"
    )
    parser.add_argument(
        "--save", "-s", help="Path to save an example frame (creates an example)"
    )
    parser.add_argument(
        "--load", "-l", help="Path to load a saved frame and print its repr"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="When saving, create a small example frame to serialize",
    )
    args = parser.parse_args()

    if args.save:
        if args.example:
            try:
                from objects.node import Node
                from objects.member import Member2D

                n1 = Node(0.0, 0.0, fixity="fixed", restrained_dofs=["x", "y", "rz"])
                n2 = Node(1.0, 0.0, fixity="fixed", restrained_dofs=["x", "y", "rz"])
                m = Member2D("beam", n1, n2, E=210e9, A=0.01, I=0.0)
                save_frame([m], args.save)
                print(f"Saved example frame to {args.save}")
            except Exception as e:
                print("Failed to create/save example frame:", e)
        else:
            print(
                "To save a frame from the CLI you must pass --example to create a demo object."
            )

    if args.load:
        try:
            obj = load_frame(args.load)
            print("Loaded object repr:")
            print(repr(obj))
            print("Loaded object type:", type(obj))
            print("Start node:", getattr(obj[0], "start_node", None))
        except Exception as e:
            print("Failed to load file:", e)
