"""
Unit tests for configuration module.

Tests Settings class validation, environment variable loading,
and singleton behavior.
"""

import os
import pytest

from app.config import Settings, get_settings, reset_settings


class TestSettings:
    """Tests for Settings class."""
    
    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
        # Clear any existing env vars
        for key in list(os.environ.keys()):
            if key.startswith("AEGIS_"):
                del os.environ[key]
    
    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("AEGIS_"):
                del os.environ[key]
    
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
    
    def test_env_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        os.environ["AEGIS_HOST"] = "127.0.0.1"
        os.environ["AEGIS_PORT"] = "9000"
        os.environ["AEGIS_DEBUG"] = "true"
        
        settings = Settings()
        
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.debug is True
    
    def test_port_validation_valid(self):
        """Test that valid port numbers are accepted."""
        os.environ["AEGIS_PORT"] = "8080"
        settings = Settings()
        assert settings.port == 8080
        
        reset_settings()
        os.environ["AEGIS_PORT"] = "1"
        settings = Settings()
        assert settings.port == 1
        
        reset_settings()
        os.environ["AEGIS_PORT"] = "65535"
        settings = Settings()
        assert settings.port == 65535
    
    def test_port_validation_invalid(self):
        """Test that invalid port numbers raise ValidationError."""
        os.environ["AEGIS_PORT"] = "0"
        with pytest.raises(ValueError):
            Settings()
        
        os.environ["AEGIS_PORT"] = "65536"
        with pytest.raises(ValueError):
            Settings()
        
        os.environ["AEGIS_PORT"] = "-1"
        with pytest.raises(ValueError):
            Settings()
    
    def test_discount_factor_validation_valid(self):
        """Test that valid discount factors are accepted."""
        os.environ["AEGIS_DISCOUNT_FACTOR"] = "0.5"
        settings = Settings()
        assert settings.discount_factor == 0.5
        
        reset_settings()
        os.environ["AEGIS_DISCOUNT_FACTOR"] = "0.0"
        settings = Settings()
        assert settings.discount_factor == 0.0
        
        reset_settings()
        os.environ["AEGIS_DISCOUNT_FACTOR"] = "1.0"
        settings = Settings()
        assert settings.discount_factor == 1.0
    
    def test_discount_factor_validation_invalid(self):
        """Test that invalid discount factors raise ValidationError."""
        os.environ["AEGIS_DISCOUNT_FACTOR"] = "-0.1"
        with pytest.raises(ValueError):
            Settings()
        
        os.environ["AEGIS_DISCOUNT_FACTOR"] = "1.1"
        with pytest.raises(ValueError):
            Settings()


class TestGetSettings:
    """Tests for get_settings singleton function."""
    
    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("AEGIS_"):
                del os.environ[key]
    
    def teardown_method(self):
        """Cleanup after each test."""
        reset_settings()
        for key in list(os.environ.keys()):
            if key.startswith("AEGIS_"):
                del os.environ[key]
    
    def test_singleton_behavior(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_reset_settings(self):
        """Test that reset_settings clears the singleton."""
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        
        # Should be different instances after reset
        assert settings1 is not settings2
    
    def test_env_changes_after_reset(self):
        """Test that environment changes are picked up after reset."""
        settings1 = get_settings()
        original_port = settings1.port
        
        os.environ["AEGIS_PORT"] = "9999"
        
        # Should still have original value
        assert get_settings().port == original_port
        
        # After reset, should pick up new value
        reset_settings()
        assert get_settings().port == 9999
