
import pytest

from src.config_loader import load_chunking_config


def test_load_chunking_config_returns_expected_values(tmp_path):
    config_file = tmp_path / "chunking.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: 100\n  overlap: 20\n",
        encoding="utf-8",
    )

    config = load_chunking_config(str(config_file))

    assert config == {
        "chunk_size": 100,
        "overlap": 20,
    }


def test_missing_config_file_raises_error(tmp_path):
    config_file = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_chunking_config(str(config_file))


@pytest.mark.parametrize(
    "yaml_content",
    [
        "[]",
        "hello",
        "",
    ],
)
def test_invalid_root_configuration_raises_error(tmp_path, yaml_content):
    config_file = tmp_path / "chunking.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration must be a YAML dictionary."):
        load_chunking_config(str(config_file))


def test_missing_chunking_section_raises_error(tmp_path):
    config_file = tmp_path / "chunking.yaml"
    config_file.write_text("other:\n  value: 10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing or invalid 'chunking' section."):
        load_chunking_config(str(config_file))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("chunk_size", '"100"'),
        ("chunk_size", "100.5"),
        ("chunk_size", "true"),
        ("overlap", '"20"'),
        ("overlap", "20.5"),
        ("overlap", "false"),
    ],
)
def test_non_integer_settings_raise_error(tmp_path, key, value):
    values = {
        "chunk_size": "100",
        "overlap": "20",
    }
    values[key] = value

    yaml_content = (
        "chunking:\n"
        f"  chunk_size: {values['chunk_size']}\n"
        f"  overlap: {values['overlap']}\n"
    )

    config_file = tmp_path / "chunking.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match=f"'{key}' must be an integer."):
        load_chunking_config(str(config_file))


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "'chunk_size' must be greater than zero."),
        (-1, 0, "'chunk_size' must be greater than zero."),
        (100, -1, "'overlap' cannot be negative."),
        (100, 100, "'overlap' must be smaller than 'chunk_size'."),
        (100, 120, "'overlap' must be smaller than 'chunk_size'."),
    ],
)
def test_invalid_chunking_values_raise_error(
    tmp_path,
    chunk_size,
    overlap,
    message,
):
    config_file = tmp_path / "chunking.yaml"
    config_file.write_text(
        f"chunking:\n  chunk_size: {chunk_size}\n  overlap: {overlap}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_chunking_config(str(config_file))

def test_invalid_yaml_syntax_raises_error(tmp_path):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: [100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid YAML configuration:"):
        load_chunking_config(str(config_file))


def test_missing_chunk_size_raises_error(tmp_path):
    config_file = tmp_path / "missing_chunk_size.yaml"
    config_file.write_text(
        "chunking:\n  overlap: 20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing chunking configuration keys:"):
        load_chunking_config(str(config_file))


def test_missing_overlap_raises_error(tmp_path):
    config_file = tmp_path / "missing_overlap.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: 100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing chunking configuration keys:"):
        load_chunking_config(str(config_file))


def test_unknown_chunking_key_raises_error(tmp_path):
    config_file = tmp_path / "unknown_key.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: 100\n  overlap: 20\n  overlapp: 10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown chunking configuration keys:"):
        load_chunking_config(str(config_file))


def test_extra_root_section_raises_error(tmp_path):
    config_file = tmp_path / "extra_section.yaml"
    config_file.write_text(
        "chunking:\n  chunk_size: 100\n  overlap: 20\n"
        "database:\n  host: localhost\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Configuration must contain only the 'chunking' section.",
    ):
        load_chunking_config(str(config_file))
