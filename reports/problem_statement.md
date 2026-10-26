# D.A.R.T.S. – Dartboard Automatic Recognition & Tracking System

*30330 Image Analysis: Image Processing – Special Assignment*

## Project Description

D.A.R.T.S. is a camera-based system that scores darts automatically from a live webcam stream. A single fixed webcam points at our own dartboard on a wall. The system finds the board, divides it into its scoring regions, detects each newly thrown dart, estimates where it hit and calculates the score.

We will test the system under different conditions: different camera angles, different lighting (a Philips Hue light strip at different intensities), and webcams of different quality.

## Problem Statement

> How can a single webcam, viewing a dartboard at an arbitrary angle, automatically locate the board, segment it into scoring regions, detect a newly thrown dart in a live video stream, and determine its score? And how is performance affected by viewing angle, lighting and camera resolution?

Key challenges:
- correcting the perspective distortion that the viewing angle causes
- telling a newly landed dart apart from other changes in the video
- estimating the tip position, which the dart itself partly hides
- separating the thin ring and sector boundaries at low resolution

## Scope

**Core goals**
1. Detect the board, correct the perspective and segment the board into scoring regions.
2. Detect a single new dart in the live stream and score it.
3. Run a basic evaluation at 2–3 camera angles, 2–3 light levels and with 2 webcams.

**Extensions**
- Multiple darts per round, including occlusion by other darts
- A full robustness study across angle, lighting and resolution
- Illumination normalisation (e.g. Retinex)
- Game logic (e.g. 501) with a live overlay
- Two cameras, to locate the dart tip better

## Planned Approach

We plan to split the system into a calibration phase, carried out while the board is empty, and a scoring phase that runs on the live video. The exact methods will be chosen through experiments. In the calibration phase, the board is first located using colour segmentation of its red and green rings combined with Canny edge detection, keeping the largest connected region as the board. When the camera views the board from an angle, the round board appears as an ellipse. We therefore plan to fit an ellipse to the outer ring, using either the Hough transform or image moments, and from it compute a transform that warps the image into a front-on view. In this corrected image we find the centre and the ring radii, detect the sector wires as straight lines with the Hough transform, and work out which sector is "20". This gives a score map in which every pixel is assigned a score from its polar coordinates (radius and angle).

In the scoring phase, a frame of the board without the new dart is stored as a reference, and each new frame is compared with it by frame differencing. A dart is considered to have landed when a small, elongated change appears and stays stable for a few frames, while large changes, such as a hand in front of the board or a change in lighting, are ignored. The difference image is then thresholded, cleaned with morphological operations and labelled to isolate the dart. The dart's direction is found from its moments, and its tip is taken as the end of the dart closest to the board. Finally, the tip position is transformed into the corrected board view and its score is looked up in the score map, after which the current frame becomes the new reference frame. To evaluate the system, we throw darts under each test condition and compare the system's score with the score read manually from the board.

## Evaluation Metrics
- board detection rate
- dart detection rate and false detections
- impact position error (mm)
- score accuracy, overall and split into sector and ring
- processing time per frame
