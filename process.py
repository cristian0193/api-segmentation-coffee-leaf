import base64
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from io import BytesIO
from pydantic import BaseModel
from ultralytics import YOLO
from matplotlib.patches import Patch
from typing import List, Dict, Sequence

class Results(BaseModel):
    image_base64: str
    classes: List[str]

# Load the model once
_model = YOLO("models/best.pt")
_DEFAULT_COLORS = ["#C7FC00", "#FF8000", "#8622FF", "#FE0056", "#00FFCE"]

def masks_segmentation(img: np.ndarray) -> Results:
    """
    Perform segmentation on the provided image using YOLO, draw smoothed contours
    color-coded by class, and return the resulting image as a Base64 string along
    with the detected class names.

    Steps:
      1. Run model inference.
      2. Build a full color palette from _DEFAULT_COLORS.
      3. Group masks into contours per class with smoothing.
      4. Draw contours onto the image.
      5. Generate a legend with color patches.
      6. Serialize the figure to JPEG in-memory and encode to Base64.

    Parameters:
      img (np.ndarray): BGR image array for segmentation.

    Returns:
      Results: Contains:
        - image_base64: Base64-encoded JPEG of the annotated image.
        - classes: List of detected class names.
    """

    # 1) Inference
    results = _model(source=img)

    # 2) Create a color palette matching the number of default colors
    palette = sns.color_palette(_DEFAULT_COLORS, len(_DEFAULT_COLORS))

    # 3) Collect all unique class IDs across results
    all_classes = sorted({
        int(c)
        for res in results
        for c in res.boxes.cls.cpu().numpy()
    })

    # 4) Map each class ID to its BGR color
    class_colors = {
        cls: tuple(int(255 * comp) for comp in palette[cls])
        for cls in all_classes
    }

    # 5) Take the first result
    res = results[0]
    img = res.orig_img.copy()

    # 6) Extract class IDs and masks
    classes = res.boxes.cls.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy().astype(bool)
    detected_classes = sorted(set(classes))

    # 7) Get smoothed contours grouped by class
    contours_by_class = get_contours_by_class(classes, masks)

    # 8) Draw contours and prepare legend
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    legend_handles = get_legend_handles(ax, class_colors, contours_by_class)

    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.8)
    ax.axis('off')
    plt.tight_layout()

    # 9) Serialize figure to JPEG in memory and encode as Base64
    buffer = BytesIO()
    fig.savefig(buffer, format='jpg', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)

    # 10) Translate class IDs to names
    class_name = [_model.names[cls] for cls in detected_classes]

    return Results(
        image_base64=img_b64,
        classes=class_name
    )


def get_legend_handles(
    ax: plt.Axes,
    class_colors: Dict[int, tuple],
    contours_by_class: Dict[int, List[np.ndarray]]
) -> List[Patch]:
    """
    Draw all contours on the given Axes and return a list of color patches
    for the legend, one per class.

    Parameters:
      ax (plt.Axes): Matplotlib axes to draw on.
      class_colors (dict): Mapping of class_id -> BGR color tuple.
      contours_by_class (dict): Mapping of class_id -> list of contour arrays.

    Returns:
       List[Patch]: Legend patches corresponding to each class.
    """
    
    legend_handles = []
    for cls, contours in contours_by_class.items():

        # Normalize BGR to 0–1 RGB for Matplotlib
        color_rgb = np.array(class_colors[cls]) / 255.0

        for cnt in contours:
            # cnt is an array of shape (N,1,2); extract x,y
            y, x = cnt[:, 0, 1], cnt[:, 0, 0]
            ax.plot(x, y, linewidth=1.5, color=color_rgb)

        # Add a patch for the legend
        legend_handles.append(Patch(facecolor=color_rgb,
                                    edgecolor='k',
                                    label=_model.names[cls]))

    return legend_handles



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