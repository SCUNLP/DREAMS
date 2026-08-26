"""Launch the local DREAMS browser demo."""

from pathlib import Path

from src.web_gateway import create_parser, run


if __name__ == "__main__":
    run(create_parser().parse_args(), Path(__file__).resolve().parents[1])
