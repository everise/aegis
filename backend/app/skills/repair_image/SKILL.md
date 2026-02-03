# repair_image

Repair or inpaint parts of an image using AI-powered restoration.

## Description

This skill repairs damaged, incomplete, or unwanted parts of images. It can fill in missing regions, remove objects, or enhance specific areas based on text prompts. Uses inpainting techniques to seamlessly blend repairs with the original image.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| image_url | string | Yes | - | URL of the image to repair |
| mask_url | string | No | null | URL of mask image (white = repair area) |
| prompt | string | Yes | - | Description of desired repair result |
| strength | number | No | 0.75 | Repair intensity (0.0-1.0) |

### Strength Parameter

- `0.0 - 0.3`: Subtle touch-ups, preserve most original content
- `0.3 - 0.6`: Moderate repairs, blend with original
- `0.6 - 0.8`: Strong repairs, significant changes allowed
- `0.8 - 1.0`: Complete regeneration of masked area

## Returns

| Field | Type | Description |
|-------|------|-------------|
| image_url | string | URL of the repaired image |
| original_url | string | URL of the original image for comparison |

## Example

### Input
```json
{
  "image_url": "https://cdn.example.com/images/original.png",
  "mask_url": "https://cdn.example.com/masks/area.png",
  "prompt": "Fill with natural grass and flowers",
  "strength": 0.75
}
```

### Output
```json
{
  "image_url": "https://cdn.example.com/images/repaired.png",
  "original_url": "https://cdn.example.com/images/original.png"
}
```

## Use Cases

1. **Object Removal**: Remove unwanted objects from images
2. **Background Replacement**: Replace or extend backgrounds
3. **Artifact Repair**: Fix compression artifacts or damage
4. **Content Extension**: Extend image boundaries (outpainting)
5. **Style Correction**: Fix style inconsistencies in generated images

## Error Codes

| Code | Description |
|------|-------------|
| INVALID_IMAGE_URL | Image URL is invalid or inaccessible |
| INVALID_MASK_URL | Mask URL is invalid or inaccessible |
| INVALID_PROMPT | Prompt is empty or invalid |
| MASK_SIZE_MISMATCH | Mask dimensions don't match image |
| TIMEOUT | Repair took too long |
| API_ERROR | Remote API returned an error |

## Rate Limits

- Maximum 10 requests per minute
- Maximum image dimension: 2048x2048
- Supported formats: PNG, JPEG, WebP

## Version

1.0.0
