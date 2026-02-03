"""
Unit tests for skill executor module.

Tests BaseSkill, concrete skill implementations, and SkillExecutor.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import httpx

from app.services.skill_executor import (
    BaseSkill,
    TextToImageSkill,
    EvaluateImageSkill,
    RepairImageSkill,
    SkillExecutor,
    SkillResult,
    SkillStatus,
    SKILL_REGISTRY,
)


class TestTextToImageSkill:
    """Tests for TextToImageSkill."""
    
    def test_validate_params_valid(self):
        """Test valid parameter validation."""
        skill = TextToImageSkill()
        
        params = skill.validate_params({
            "prompt": "A beautiful landscape",
            "width": 1024,
            "height": 768,
        })
        
        assert params["prompt"] == "A beautiful landscape"
        assert params["width"] == 1024
        assert params["height"] == 768
        assert params["steps"] == 20  # default
    
    def test_validate_params_missing_prompt(self):
        """Test validation fails without prompt."""
        skill = TextToImageSkill()
        
        with pytest.raises(ValueError, match="prompt is required"):
            skill.validate_params({})
    
    def test_validate_params_empty_prompt(self):
        """Test validation fails with empty prompt."""
        skill = TextToImageSkill()
        
        with pytest.raises(ValueError, match="prompt is required"):
            skill.validate_params({"prompt": ""})
    
    def test_validate_params_defaults(self):
        """Test default values are applied."""
        skill = TextToImageSkill()
        
        params = skill.validate_params({"prompt": "test"})
        
        assert params["width"] == 512
        assert params["height"] == 512
        assert params["steps"] == 20
        assert params["negative_prompt"] is None
        assert params["seed"] is None
    
    def test_format_result(self):
        """Test result formatting."""
        skill = TextToImageSkill()
        
        raw = {
            "image_url": "http://example.com/img.png",
            "width": 512,
            "height": 512,
            "seed": 42,
            "extra_field": "ignored",
        }
        
        formatted = skill.format_result(raw)
        
        assert formatted["image_url"] == "http://example.com/img.png"
        assert formatted["width"] == 512
        assert formatted["seed"] == 42
        assert "extra_field" not in formatted


class TestEvaluateImageSkill:
    """Tests for EvaluateImageSkill."""
    
    def test_validate_params_valid(self):
        """Test valid parameter validation."""
        skill = EvaluateImageSkill()
        
        params = skill.validate_params({
            "image_url": "http://example.com/img.png",
            "criteria": ["quality"],
        })
        
        assert params["image_url"] == "http://example.com/img.png"
        assert params["criteria"] == ["quality"]
    
    def test_validate_params_missing_url(self):
        """Test validation fails without image_url."""
        skill = EvaluateImageSkill()
        
        with pytest.raises(ValueError, match="image_url is required"):
            skill.validate_params({})
    
    def test_validate_params_default_criteria(self):
        """Test default criteria are applied."""
        skill = EvaluateImageSkill()
        
        params = skill.validate_params({"image_url": "http://example.com/img.png"})
        
        assert "quality" in params["criteria"]
        assert "aesthetics" in params["criteria"]
    
    def test_format_result(self):
        """Test result formatting."""
        skill = EvaluateImageSkill()
        
        raw = {
            "scores": {"quality": 0.85},
            "overall_score": 0.85,
            "feedback": "Good quality",
        }
        
        formatted = skill.format_result(raw)
        
        assert formatted["scores"] == {"quality": 0.85}
        assert formatted["overall_score"] == 0.85
        assert formatted["feedback"] == "Good quality"


class TestRepairImageSkill:
    """Tests for RepairImageSkill."""
    
    def test_validate_params_valid(self):
        """Test valid parameter validation."""
        skill = RepairImageSkill()
        
        params = skill.validate_params({
            "image_url": "http://example.com/img.png",
            "prompt": "Fix the background",
            "strength": 0.8,
        })
        
        assert params["image_url"] == "http://example.com/img.png"
        assert params["prompt"] == "Fix the background"
        assert params["strength"] == 0.8
    
    def test_validate_params_missing_image_url(self):
        """Test validation fails without image_url."""
        skill = RepairImageSkill()
        
        with pytest.raises(ValueError, match="image_url is required"):
            skill.validate_params({"prompt": "test"})
    
    def test_validate_params_missing_prompt(self):
        """Test validation fails without prompt."""
        skill = RepairImageSkill()
        
        with pytest.raises(ValueError, match="prompt is required"):
            skill.validate_params({"image_url": "http://example.com/img.png"})
    
    def test_validate_params_defaults(self):
        """Test default values are applied."""
        skill = RepairImageSkill()
        
        params = skill.validate_params({
            "image_url": "http://example.com/img.png",
            "prompt": "test",
        })
        
        assert params["strength"] == 0.75
        assert params["mask_url"] is None


class TestSkillSubmitPoll:
    """Tests for skill submit-poll pattern."""
    
    @pytest_asyncio.fixture
    async def mock_client(self):
        """Create a mock HTTP client."""
        client = AsyncMock(spec=httpx.AsyncClient)
        return client
    
    @pytest.mark.asyncio
    async def test_submit_success(self, mock_client):
        """Test successful task submission."""
        skill = TextToImageSkill(http_client=mock_client)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "task-123"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        
        task_id = await skill.submit({"prompt": "test"})
        
        assert task_id == "task-123"
        mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_poll_success(self, mock_client):
        """Test successful task polling."""
        skill = TextToImageSkill(http_client=mock_client)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "completed",
            "result": {"image_url": "http://example.com/img.png"},
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response
        
        result = await skill.poll("task-123")
        
        assert result["status"] == "completed"
        mock_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_complete_flow(self, mock_client):
        """Test complete execute flow with submit and poll."""
        skill = TextToImageSkill(http_client=mock_client)
        skill.poll_interval = 0.01  # Fast polling for tests
        
        # Mock submit response
        submit_response = MagicMock()
        submit_response.json.return_value = {"task_id": "task-123"}
        submit_response.raise_for_status = MagicMock()
        
        # Mock poll response (completed immediately)
        poll_response = MagicMock()
        poll_response.json.return_value = {
            "status": "completed",
            "result": {
                "image_url": "http://example.com/img.png",
                "width": 512,
                "height": 512,
                "seed": 42,
            },
        }
        poll_response.raise_for_status = MagicMock()
        
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = poll_response
        
        result = await skill.execute({"prompt": "test image"})
        
        assert result.status == SkillStatus.COMPLETED
        assert result.task_id == "task-123"
        assert result.result["image_url"] == "http://example.com/img.png"
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_execute_poll_pending_then_complete(self, mock_client):
        """Test polling with pending status before completion."""
        skill = TextToImageSkill(http_client=mock_client)
        skill.poll_interval = 0.01
        
        submit_response = MagicMock()
        submit_response.json.return_value = {"task_id": "task-123"}
        submit_response.raise_for_status = MagicMock()
        
        # First poll: pending, second poll: completed
        pending_response = MagicMock()
        pending_response.json.return_value = {"status": "pending", "progress": 0.5}
        pending_response.raise_for_status = MagicMock()
        
        completed_response = MagicMock()
        completed_response.json.return_value = {
            "status": "completed",
            "result": {"image_url": "http://example.com/img.png"},
        }
        completed_response.raise_for_status = MagicMock()
        
        mock_client.post.return_value = submit_response
        mock_client.get.side_effect = [pending_response, completed_response]
        
        result = await skill.execute({"prompt": "test"})
        
        assert result.status == SkillStatus.COMPLETED
        assert result.poll_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_task_failed(self, mock_client):
        """Test handling of failed tasks."""
        skill = TextToImageSkill(http_client=mock_client)
        skill.poll_interval = 0.01
        
        submit_response = MagicMock()
        submit_response.json.return_value = {"task_id": "task-123"}
        submit_response.raise_for_status = MagicMock()
        
        failed_response = MagicMock()
        failed_response.json.return_value = {
            "status": "failed",
            "error": "Generation failed",
        }
        failed_response.raise_for_status = MagicMock()
        
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = failed_response
        
        result = await skill.execute({"prompt": "test"})
        
        assert result.status == SkillStatus.FAILED
        assert "failed" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_validation_error(self, mock_client):
        """Test handling of validation errors."""
        skill = TextToImageSkill(http_client=mock_client)
        
        result = await skill.execute({})  # Missing prompt
        
        assert result.status == SkillStatus.FAILED
        assert "prompt" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_http_error(self, mock_client):
        """Test handling of HTTP errors."""
        skill = TextToImageSkill(http_client=mock_client)
        
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="Internal error"),
        )
        
        result = await skill.execute({"prompt": "test"})
        
        assert result.status == SkillStatus.FAILED
        assert "HTTP error" in result.error


class TestSkillExecutor:
    """Tests for SkillExecutor."""
    
    def test_list_skills(self):
        """Test listing available skills."""
        executor = SkillExecutor()
        
        skills = executor.list_skills()
        
        assert len(skills) == 3
        skill_names = [s["name"] for s in skills]
        assert "text_to_image" in skill_names
        assert "evaluate_image" in skill_names
        assert "repair_image" in skill_names
    
    def test_get_skill_valid(self):
        """Test getting a valid skill."""
        executor = SkillExecutor()
        
        skill = executor.get_skill("text_to_image")
        
        assert isinstance(skill, TextToImageSkill)
    
    def test_get_skill_invalid(self):
        """Test getting an invalid skill raises error."""
        executor = SkillExecutor()
        
        with pytest.raises(ValueError, match="Unknown skill"):
            executor.get_skill("invalid_skill")
    
    def test_get_skill_caching(self):
        """Test that skill instances are cached."""
        executor = SkillExecutor()
        
        skill1 = executor.get_skill("text_to_image")
        skill2 = executor.get_skill("text_to_image")
        
        assert skill1 is skill2
    
    @pytest.mark.asyncio
    async def test_execute_delegates_to_skill(self):
        """Test that execute delegates to the correct skill."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        executor = SkillExecutor(http_client=mock_client)
        
        # Mock responses
        submit_response = MagicMock()
        submit_response.json.return_value = {"task_id": "task-123"}
        submit_response.raise_for_status = MagicMock()
        
        poll_response = MagicMock()
        poll_response.json.return_value = {
            "status": "completed",
            "result": {"image_url": "http://example.com/img.png"},
        }
        poll_response.raise_for_status = MagicMock()
        
        mock_client.post.return_value = submit_response
        mock_client.get.return_value = poll_response
        
        # Reduce poll interval for faster test
        skill = executor.get_skill("text_to_image")
        skill.poll_interval = 0.01
        
        result = await executor.execute("text_to_image", {"prompt": "test"})
        
        assert result.skill_name == "text_to_image"
        assert result.status == SkillStatus.COMPLETED


class TestSkillRegistry:
    """Tests for skill registry."""
    
    def test_all_skills_registered(self):
        """Test that all expected skills are registered."""
        expected_skills = ["text_to_image", "evaluate_image", "repair_image"]
        
        for skill_name in expected_skills:
            assert skill_name in SKILL_REGISTRY
    
    def test_registered_skills_are_valid(self):
        """Test that all registered skills are valid BaseSkill subclasses."""
        for name, skill_class in SKILL_REGISTRY.items():
            assert issubclass(skill_class, BaseSkill)
            assert skill_class.name == name
            assert skill_class.description
            assert skill_class.submit_endpoint
