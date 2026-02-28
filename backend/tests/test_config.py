"""
Unit tests for configuration module.

Tests Settings class validation, YAML file loading,
and singleton behavior.
"""

import os
import textwrap
import pytest

from app.config import Settings, get_settings, reset_settings


class TestSettings:
    """Tests for Settings class."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        os.environ.pop("AEGIS_CONFIG", None)

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = Settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.debug is False
        assert "sqlite+aiosqlite" in settings.database_url
        assert settings.max_trajectory_steps == 10
        assert settings.replay_buffer_size == 10000
        assert settings.training_batch_size == 32
        assert settings.discount_factor == 0.99

    def test_yaml_loading(self, tmp_path):
        """Test that values are loaded from a YAML config file."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text(textwrap.dedent("""\
            server:
              host: "127.0.0.1"
              port: 9000
              debug: true
            database:
              url: "sqlite+aiosqlite:///./test.db"
        """))

        os.environ["AEGIS_CONFIG"] = str(cfg)
        settings = get_settings()

        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.debug is True
        assert settings.database_url == "sqlite+aiosqlite:///./test.db"
        # Unset values should keep defaults
        assert settings.training_batch_size == 32

    def test_partial_yaml(self, tmp_path):
        """Test that a partial YAML file fills only specified fields."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text("server:\n  port: 4000\n")

        os.environ["AEGIS_CONFIG"] = str(cfg)
        settings = get_settings()

        assert settings.port == 4000
        assert settings.host == "0.0.0.0"  # default

    def test_missing_yaml_uses_defaults(self, tmp_path):
        """Test that a missing YAML file gracefully uses all defaults."""
        os.environ["AEGIS_CONFIG"] = str(tmp_path / "nonexistent.yaml")
        settings = get_settings()

        assert settings.port == 8000
        assert settings.debug is False

    def test_port_validation_valid(self):
        """Test that valid port numbers are accepted."""
        assert Settings(port=8080).port == 8080
        assert Settings(port=1).port == 1
        assert Settings(port=65535).port == 65535

    def test_port_validation_invalid(self):
        """Test that invalid port numbers raise ValidationError."""
        with pytest.raises(ValueError):
            Settings(port=0)
        with pytest.raises(ValueError):
            Settings(port=65536)
        with pytest.raises(ValueError):
            Settings(port=-1)

    def test_discount_factor_validation_valid(self):
        """Test that valid discount factors are accepted."""
        assert Settings(discount_factor=0.5).discount_factor == 0.5
        assert Settings(discount_factor=0.0).discount_factor == 0.0
        assert Settings(discount_factor=1.0).discount_factor == 1.0

    def test_discount_factor_validation_invalid(self):
        """Test that invalid discount factors raise ValidationError."""
        with pytest.raises(ValueError):
            Settings(discount_factor=-0.1)
        with pytest.raises(ValueError):
            Settings(discount_factor=1.1)

    def test_memory_fields(self, tmp_path):
        """Test memory-related settings from YAML."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text(textwrap.dedent("""\
            memory:
              max_tokens: 8000
              compress_on_add: false
              protected_pairs: 5
        """))

        os.environ["AEGIS_CONFIG"] = str(cfg)
        settings = get_settings()

        assert settings.memory_max_tokens == 8000
        assert settings.memory_compress_on_add is False
        assert settings.memory_protected_pairs == 5
        # Unset memory field keeps default
        assert settings.memory_compress_on_get is True

    def test_vector_fields(self, tmp_path):
        """Test vector-related settings from YAML."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text(textwrap.dedent("""\
            vector:
              chroma_persist_dir: "/custom/chroma"
              max_context_tokens: 64000
        """))

        os.environ["AEGIS_CONFIG"] = str(cfg)
        settings = get_settings()

        assert settings.chroma_persist_dir == "/custom/chroma"
        assert settings.max_context_tokens == 64000


class TestGetSettings:
    """Tests for get_settings singleton function."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        os.environ.pop("AEGIS_CONFIG", None)

    def test_singleton_behavior(self, tmp_path):
        """Test that get_settings returns the same instance."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text("server:\n  port: 8000\n")
        os.environ["AEGIS_CONFIG"] = str(cfg)

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reset_settings(self, tmp_path):
        """Test that reset_settings clears the singleton."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text("server:\n  port: 8000\n")
        os.environ["AEGIS_CONFIG"] = str(cfg)

        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()

        assert settings1 is not settings2

    def test_config_changes_after_reset(self, tmp_path):
        """Test that YAML changes are picked up after reset."""
        cfg = tmp_path / "aegis.yaml"
        cfg.write_text("server:\n  port: 8000\n")
        os.environ["AEGIS_CONFIG"] = str(cfg)

        settings1 = get_settings()
        assert settings1.port == 8000

        # Modify the config file
        cfg.write_text("server:\n  port: 9999\n")

        # Should still have original value (cached)
        assert get_settings().port == 8000

        # After reset, should pick up new value
        reset_settings()
        assert get_settings().port == 9999

    def test_explicit_config_path(self, tmp_path):
        """Test passing config_path directly to get_settings."""
        cfg = tmp_path / "custom.yaml"
        cfg.write_text("server:\n  debug: true\n")

        settings = get_settings(config_path=str(cfg))
        assert settings.debug is True
