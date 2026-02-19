# from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# model_name = "microsoft/trocr-base-printed"

# print("Downloading model...")
# model = VisionEncoderDecoderModel.from_pretrained(model_name)
# processor = TrOCRProcessor.from_pretrained(model_name)

# print("Saving locally...")
# model.save_pretrained("./trocr-base-printed")
# processor.save_pretrained("./trocr-base-printed")

# print("Done.")


import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Ensure offline mode (no internet)
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Path to your downloaded model
model_path = "./trocr-base-printed"

# Load model & processor
model = VisionEncoderDecoderModel.from_pretrained(
    model_path,
    local_files_only=True
)

processor = TrOCRProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False
)

# Load a sample image
image_path = "scanned_output.jpg"
# image_path = "sample_image.png"  # replace with your image path
image = Image.open(image_path).convert("RGB")

# Preprocess image
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate OCR prediction
generated_ids = model.generate(pixel_values)

# Decode text
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("OCR result:", text)
