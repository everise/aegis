# text_to_image

Generate images from text prompts using AI image generation models.

## Description

This skill converts natural language descriptions into high-quality images. It supports various parameters to control the output, including image dimensions, generation steps, and seed for reproducibility.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| prompt | string | Yes | - | Text description of the image to generate |
| negative_prompt | string | No | null | What to avoid in the generated image |
| width | integer | No | 512 | Image width (64-2048) |
| height | integer | No | 512 | Image height (64-2048) |
| steps | integer | No | 20 | Number of diffusion steps (1-100) |
| seed | integer | No | random | Seed for reproducible generation |

## Returns

| Field | Type | Description |
|-------|------|-------------|
| image_url | string | URL of the generated image |
| width | integer | Actual width of generated image |
| height | integer | Actual height of generated image |
| seed | integer | Seed used for generation |

## Example

### Input
```json
{
  "prompt": "A beautiful sunset over mountains, digital art style",
  "negative_prompt": "blurry, low quality",
  "width": 1024,
  "height": 768,
  "steps": 30
}
```

### Output
```json
{
  "image_url": "https://cdn.example.com/images/abc123.png",
  "width": 1024,
  "height": 768,
  "seed": 42
}
```

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_PROMPT | Prompt is empty or invalid |
| DIMENSION_ERROR | Width or height out of valid range |
| TIMEOUT | Generation took too long |
| API_ERROR | Remote API returned an error |

## Rate Limits

- Maximum 10 requests per minute
- Maximum image dimension: 2048x2048

## Version

1.0.0
