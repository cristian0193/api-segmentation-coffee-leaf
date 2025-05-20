from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import process
from typing import List

app = FastAPI(title="API de Base64")

port = int(os.environ.get("PORT", 8000))
app.run("main:app", host="0.0.0.0", port=port)

class SegmentationRequest(BaseModel):
    image_base64: str

class SegmentationResponse(BaseModel):
    image_base64: str
    classes: List[str]

@app.post("/segmentation", response_model=SegmentationResponse, status_code=200)
def decode_image(req: SegmentationRequest):
    """
    Decode a Base64-encoded image, perform segmentation, and return the
    segmented image along with the detected class names.

    Processing steps:
      1. Decode the input Base64 string into raw bytes.
      2. Convert raw bytes to a NumPy array and then to an OpenCV BGR image.
      3. Pass the image to the segmentation function `masks_segmentation`.
      4. Return the resulting image (as Base64) and the list of classes.

    Args:
        req (SegmentationRequest): Payload containing the Base64 image.

    Returns:
        SegmentationResponse: Contains the segmented image and class list.

    Raises:
        HTTPException 400: If Base64 decoding fails or image cannot be decoded.
        HTTPException 500: If any other error occurs during processing.
    """
    try:
        # Step 1: Base64 → bytes
        try:
            img_data = base64.b64decode(req.image_base64)
        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Base64 invalid")

        # Step 2: bytes → NumPy array → OpenCV image
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Image could not be decoded")

        # Step 3: Perform segmentation
        result = process.masks_segmentation(img)

        # Step 4: Build and return the response
        return SegmentationResponse(
            image_base64=result.image_base64,
            classes=result.classes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

