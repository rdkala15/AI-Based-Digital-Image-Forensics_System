"""
Copy-Move Forgery Detection

Detects potentially duplicated regions within the same image
using local feature matching.

Important:
This is a forensic indicator, not definitive proof of manipulation.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw


def detect_copy_move(
    image,
    min_matches=8,
    min_distance=40
):
    """
    Detect potentially duplicated regions within the same image.

    Parameters
    ----------
    image : PIL.Image
        Input image.

    min_matches : int
        Minimum number of feature matches required.

    min_distance : int
        Minimum spatial distance between matched points.
        This helps avoid matching a feature with itself or
        a nearby naturally occurring feature.

    Returns
    -------
    result : dict
        Detection results and visualization.
    """

    # Convert PIL image to RGB
    image_rgb = image.convert("RGB")

    # Convert PIL -> OpenCV
    image_array = np.array(image_rgb)

    # RGB -> grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8
    )

    # Detect features
    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    # Default visualization
    visualization = image_rgb.copy()

    if descriptors is None or len(keypoints) < 10:
        return {
            "detected": False,
            "match_count": 0,
            "visualization": visualization,
            "findings": [
                "Not enough local features were detected for copy-move analysis."
            ]
        }

    # Brute-force matcher
    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    # Match every descriptor against descriptors
    # from the same image.
    matches = matcher.knnMatch(
        descriptors,
        descriptors,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        # Ignore exact self-match
        if m.queryIdx == m.trainIdx:
            continue

        # Lowe ratio test
        if m.distance < 0.75 * n.distance:

            point1 = np.array(
                keypoints[m.queryIdx].pt
            )

            point2 = np.array(
                keypoints[m.trainIdx].pt
            )

            spatial_distance = np.linalg.norm(
                point1 - point2
            )

            # Ignore matches that are too close
            if spatial_distance >= min_distance:

                good_matches.append(m)

    # Remove duplicate reverse matches
    unique_pairs = set()
    filtered_matches = []

    for match in good_matches:

        pair = tuple(
            sorted(
                [match.queryIdx, match.trainIdx]
            )
        )

        if pair not in unique_pairs:
            unique_pairs.add(pair)
            filtered_matches.append(match)

    good_matches = filtered_matches

    match_count = len(good_matches)

    # Create visualization
    vis_array = np.array(image_rgb).copy()

    # Draw matched feature points
    for match in good_matches:

        p1 = tuple(
            map(
                int,
                keypoints[match.queryIdx].pt
            )
        )

        p2 = tuple(
            map(
                int,
                keypoints[match.trainIdx].pt
            )
        )

        cv2.circle(
            vis_array,
            p1,
            6,
            (255, 0, 0),
            2
        )

        cv2.circle(
            vis_array,
            p2,
            6,
            (255, 0, 0),
            2
        )

        cv2.line(
            vis_array,
            p1,
            p2,
            (255, 0, 0),
            1
        )

    visualization = Image.fromarray(vis_array)

    # Determine finding
    detected = match_count >= min_matches

    findings = []

    if detected:

        findings.append(
            f"{match_count} potentially duplicated "
            "feature correspondences were detected."
        )

        findings.append(
            "The highlighted regions contain visually "
            "similar local features within the same image."
        )

        findings.append(
            "These regions require further examination "
            "for possible copy-move manipulation."
        )

    else:

        findings.append(
            "No strong copy-move pattern was detected "
            "using the current feature-matching settings."
        )

    findings.append(
        "Copy-move analysis is a forensic indicator and "
        "does not independently prove image manipulation."
    )

    return {
        "detected": detected,
        "match_count": match_count,
        "visualization": visualization,
        "findings": findings
    }


if __name__ == "__main__":

    print("=" * 60)
    print("COPY-MOVE FORENSICS MODULE TEST")
    print("=" * 60)

    # Create a simple test image
    test_image = Image.new(
        "RGB",
        (600, 400),
        "white"
    )

    draw = ImageDraw.Draw(test_image)

    # Original object
    draw.rectangle(
        (100, 100, 200, 200),
        fill="black"
    )

    draw.ellipse(
        (125, 125, 175, 175),
        fill="white"
    )

    # Copied object
    draw.rectangle(
        (350, 200, 450, 300),
        fill="black"
    )

    draw.ellipse(
        (375, 225, 425, 275),
        fill="white"
    )

    result = detect_copy_move(
        test_image
    )

    print("\nFeature matches:")
    print(result["match_count"])

    print("\nCopy-move detected:")
    print(result["detected"])

    print("\nFindings:")

    for finding in result["findings"]:
        print("-", finding)

    print("\nCOPY-MOVE TEST COMPLETED")
    print("=" * 60)