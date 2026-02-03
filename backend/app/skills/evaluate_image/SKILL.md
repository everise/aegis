# evaluate_image

Evaluate the quality and aesthetics of images using AI analysis.

## Description

This skill analyzes images and provides quality scores across multiple criteria. It can assess overall image quality, aesthetic appeal, and how well the image aligns with a given prompt. Useful for quality control in image generation pipelines.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| image_url | string | Yes | - | URL of the image to evaluate |
| criteria | array[string] | No | ["quality", "aesthetics", "prompt_alignment"] | Evaluation criteria to assess |

### Available Criteria

- `quality`: Technical quality (sharpness, noise, artifacts)
- `aesthetics`: Visual appeal and composition
- `prompt_alignment`: How well image matches intended description
- `coherence`: Internal consistency of the image
- `creativity`: Uniqueness and artistic value

## Returns

| Field | Type | Description |
|-------|------|-------------|
| scores | object | Individual scores for each criterion (0.0-1.0) |
| overall_score | number | Weighted average of all scores (0.0-1.0) |
| feedback | string | Human-readable feedback about the image |

## Example

### Input
```json
{
  "image_url": "https://cdn.example.com/images/abc123.png",
  "criteria": ["quality", "aesthetics", "prompt_alignment"]
}
```

### Output
```json
{
  "scores": {
    "quality": 0.85,
    "aesthetics": 0.78,
    "prompt_alignment": 0.92
  },
  "overall_score": 0.85,
  "feedback": "Image meets quality standards. Good technical quality with strong prompt alignment."
}
```

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0 | Excellent |
| 0.7 - 0.9 | Good |
| 0.5 - 0.7 | Acceptable |
| 0.3 - 0.5 | Poor |
| 0.0 - 0.3 | Unacceptable |

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_URL | Image URL is invalid or inaccessible |
| INVALID_CRITERIA | Unknown evaluation criteria specified |
| TIMEOUT | Evaluation took too long |
| API_ERROR | Remote API returned an error |

## Rate Limits

- Maximum 30 requests per minute
- Supported image formats: PNG, JPEG, WebP

## Version

1.0.0
