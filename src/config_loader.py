
from pathlib import Path

import yaml


def load_chunking_config(
    config_path: str = "configs/chunking.yaml",
) -> dict[str, int]:
    path = Path(config_path)

    with path.open("r", encoding="utf-8-sig") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Invalid YAML configuration: {error}"
            ) from error

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML dictionary.")

    if "chunking" not in config:
        raise ValueError("Missing or invalid 'chunking' section.")

    if set(config.keys()) != {"chunking"}:
        raise ValueError(
            "Configuration must contain only the 'chunking' section."
        )

    chunking = config["chunking"]

    if not isinstance(chunking, dict):
        raise ValueError("'chunking' section must be a dictionary.")

    allowed_keys = {"chunk_size", "overlap"}
    unknown_keys = set(chunking.keys()) - allowed_keys

    if unknown_keys:
        raise ValueError(
            f"Unknown chunking configuration keys: {sorted(unknown_keys)}"
        )

    missing_keys = allowed_keys - set(chunking.keys())

    if missing_keys:
        raise ValueError(
            f"Missing chunking configuration keys: {sorted(missing_keys)}"
        )

    chunk_size = chunking["chunk_size"]
    overlap = chunking["overlap"]

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("'chunk_size' must be an integer.")

    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError("'overlap' must be an integer.")

    if chunk_size <= 0:
        raise ValueError("'chunk_size' must be greater than zero.")

    if overlap < 0:
        raise ValueError("'overlap' cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("'overlap' must be smaller than 'chunk_size'.")

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
    }


if __name__ == "__main__":
    settings = load_chunking_config()
    print("Chunking configuration loaded successfully!")
    print(settings)
