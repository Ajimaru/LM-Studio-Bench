# Image samples for vision testing

Due to file size constraints, repository versions may omit image binaries.
For local testing, place sample images in this directory.

Current local test files (renamed to avoid label leakage in URL/file names):

Recommended test images:

1. `test_image_001.jpg` - Simple object (cat)
2. `test_image_002.jpg` - Multiple objects (street scene)
3. `test_image_003.jpg` - Text content (STOP sign)

Source and licensing documentation:

- `LICENSES.md` - attribution and license details for each image
- `wikimedia_manifest.json` - raw metadata captured during download

If you need additional examples, use public URLs from datasets like:

- COCO validation set
- Flickr8k
- Visual Genome

Example VQA test cases:

- Image: Cat on a couch
  Question: What animal is in the image?
  Answer: cat

- Image: Street scene with cars
  Question: Name two object types visible in this image.
  Answer: e.g. building, car

- Image: STOP sign
  Question: What text is written on the sign?
  Answer: stop
