import base64
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from typing import List, Dict, Sequence

app = FastAPI(title="API de Base64")


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
        result = masks_segmentation(img)

        # Step 4: Build and return the response
        return SegmentationResponse(
            image_base64=result.image_base64,
            classes=result.classes
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Results(BaseModel):
    image_base64: str
    classes: List[str]


# Load the model once
_model = YOLO("models/best.pt")
_DEFAULT_COLORS = [
    (0, 252, 199),  # #C7FC00 -> BGR
    (0, 128, 255),  # #FF8000 -> BGR
    (255, 34, 134),  # #8622FF -> BGR
    (86, 0, 254),  # #FE0056 -> BGR
    (206, 255, 0)  # #00FFCE -> BGR
]


def masks_segmentation(img: np.ndarray) -> Results:
    """
    Perform segmentation on the provided image using YOLO, draw smoothed contours
    color-coded by class, and return the resulting image as a Base64 string along
    with the detected class names.

    Steps:
      1. Run model inference.
      2. Group masks into contours per class.
      3. Draw contours onto the image using OpenCV.
      4. Add a legend with color rectangles and text.
      5. Encode the image to JPEG and then to Base64.

    Parameters:
      img (np.ndarray): BGR image array for segmentation.

    Returns:
      Results: Contains:
        - image_base64: Base64-encoded JPEG of the annotated image.
        - classes: List of detected class names.
    """

    # 1) Inference
    results = _model(source=img)

    # 2) Take the first result
    res = results[0]
    img_annotated = res.orig_img.copy()

    # 3) Extract class IDs and masks
    classes = res.boxes.cls.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy().astype(bool)
    detected_classes = sorted(set(classes))

    # 4) Create color mapping for detected classes
    class_colors = {
        cls: _DEFAULT_COLORS[cls % len(_DEFAULT_COLORS)]
        for cls in detected_classes
    }

    # 5) Get contours grouped by class
    contours_by_class = get_contours_by_class(classes, masks)

    # 6) Draw contours on the image
    draw_contours_on_image(img_annotated, contours_by_class, class_colors)

    # 7) Add legend to the image
    img_with_legend = add_legend_to_image(img_annotated, detected_classes, class_colors)

    # 8) Encode image to Base64
    img_b64 = encode_image_to_base64(img_with_legend)

    # 9) Translate class IDs to names
    class_names = [_model.names[cls] for cls in detected_classes]

    return Results(
        image_base64=img_b64,
        classes=class_names
    )


def draw_contours_on_image(
        img: np.ndarray,
        contours_by_class: Dict[int, List[np.ndarray]],
        class_colors: Dict[int, tuple]
) -> None:
    """
    Draw all contours on the image using OpenCV.

    Parameters:
      img (np.ndarray): Image to draw on (modified in place).
      contours_by_class (dict): Mapping of class_id -> list of contour arrays.
      class_colors (dict): Mapping of class_id -> BGR color tuple.
    """
    for cls, contours in contours_by_class.items():
        color = class_colors[cls]
        for contour in contours:
            # Draw contour with specified color and thickness
            cv2.drawContours(img, [contour], -1, color, thickness=2)


def add_legend_to_image(
        img: np.ndarray,
        detected_classes: List[int],
        class_colors: Dict[int, tuple]
) -> np.ndarray:
    """
    Add a legend to the image showing class names and their corresponding colors.

    Parameters:
      img (np.ndarray): Original image.
      detected_classes (list): List of detected class IDs.
      class_colors (dict): Mapping of class_id -> BGR color tuple.

    Returns:
      np.ndarray: Image with legend added.
    """
    img_height, img_width = img.shape[:2]

    # Legend dimensions
    legend_width = 200
    legend_item_height = 30
    legend_height = len(detected_classes) * legend_item_height + 20

    # Create a new image with space for the legend
    new_width = img_width + legend_width
    new_img = np.ones((max(img_height, legend_height), new_width, 3), dtype=np.uint8) * 255

    # Copy original image to the left side
    new_img[:img_height, :img_width] = img

    # Add legend background
    legend_x_start = img_width + 10
    legend_y_start = 10

    # Draw legend items
    for i, cls in enumerate(detected_classes):
        y_pos = legend_y_start + i * legend_item_height
        color = class_colors[cls]
        class_name = _model.names[cls]

        # Draw color rectangle
        rect_start = (legend_x_start, y_pos)
        rect_end = (legend_x_start + 20, y_pos + 20)
        cv2.rectangle(new_img, rect_start, rect_end, color, -1)
        cv2.rectangle(new_img, rect_start, rect_end, (0, 0, 0), 1)

        # Add text
        text_pos = (legend_x_start + 30, y_pos + 15)
        cv2.putText(new_img, class_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return new_img


def encode_image_to_base64(img: np.ndarray) -> str:
    """
    Encode an OpenCV image to Base64 string.

    Parameters:
      img (np.ndarray): Image to encode.

    Returns:
      str: Base64-encoded image string.
    """
    # Encode image as JPEG
    success, img_encoded = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Failed to encode image")

    # Convert to Base64
    img_bytes = img_encoded.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_b64


def get_contours_by_class(
        classes: Sequence[int],
        masks: Sequence[np.ndarray]
) -> Dict[int, List[np.ndarray]]:
    """
    Extract and group contours from binary masks by their corresponding class IDs.

    This function iterates over each pair of class ID and binary mask, converts
    the mask to an 8-bit format, finds all external contours, and aggregates
    them into a dictionary keyed by class ID.

    Parameters:
       classes (Sequence[int]):
         An iterable of integer class IDs, one for each mask in `masks`.
       masks (Sequence[np.ndarray]):
         An iterable of 2D boolean or integer arrays representing binary masks.
         Each mask should have values of 0 or 1 (or False/True).

    Returns:
       Dict[int, List[np.ndarray]]:
          A mapping from each class ID to a list of contour arrays. Each contour
          is a NumPy array of shape (N, 1, 2), where N is the number of points.
    """
    contours_by_class = {}
    for cls, mask in zip(classes, masks):
        # Convert boolean or 0/1 mask to uint8 format with values 0 or 255
        mask_uint8 = (mask.astype(np.uint8) * 255)

        # Find all external contours without approximation
        contours, _ = cv2.findContours(mask_uint8,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        # Append found contours to the list for this class ID
        contours_by_class.setdefault(int(cls), []).extend(contours)

    return contours_by_class