# Soccer Player Re-Identification Report

This report outlines the approach, methodology, and outcomes of experiments conducted to perform object tracking and re-identification (ReID) on a video dataset, focusing on tracking players, referees, and a ball in a sports video. The task utilized advanced tracking algorithms and ReID techniques to maintain consistent identities across frames. Below, I detail the methodologies explored, the techniques attempted, their outcomes, challenges faced, and future steps to improve performance.

## Approach and Methodology

The objective was to track multiple objects (players, referees, and a ball) in a 15-second 720p video (`15sec_input_720p.mp4`) and assign consistent identities using ReID techniques. The approach combined state-of-the-art object detection with tracking and ReID algorithms, leveraging the YOLO model for detection and various trackers for maintaining object identities. The methodology involved the following steps:

1. **Object Detection**:

   - Utilized YOLOv11 (pretrained models `best.pt` and `best (2).pt`) for detecting objects in each frame.
   - YOLOv11 provided bounding box coordinates, class labels, and confidence scores for objects like players, referees, and the ball.

2. **Tracking**:

   - Employed tracking algorithms to associate detections across frames, ensuring objects retained consistent track IDs.
   - Tested multiple trackers: BoT-SORT, ByteTrack, Deep SORT, and StrongSORT, with configurations like `botsort.yaml` for BoT-SORT.

3. **Re-Identification (ReID)**:

   - Integrated ReID to handle occlusions and reassign correct track IDs when objects reappeared.
   - Used feature extractors (OSNet and ResNet50) from the `torchreid` library to generate embeddings for each detected object.
   - Applied cosine similarity (threshold 0.7) to match embeddings across frames, maintaining identity consistency.

4. **Output Generation**:

   - Annotated frames with bounding boxes, class labels, and track IDs.
   - Saved outputs as AVI and MP4 files (`output_with_osnet3.mp4`, `output_with_resnet.mp4`) with a sample frame displayed for verification.

The experiments were conducted in Google Colab with GPU support to leverage CUDA for faster processing. The setup involved cloning relevant repositories, installing dependencies, and downloading pretrained model weights.

## Techniques Tried and Outcomes

Several tracking and ReID algorithms were researched and tested, including PRTreID, BoT-SORT, ByteTrack, Deep SORT, OC-SORT, and StrongSORT. Below is a detailed account of the techniques attempted and their outcomes.

### 1. BoT-SORT

**Description**:

- BoT-SORT (ByteTrack with StrongSORT enhancements) is a tracking algorithm that integrates YOLO detections with robust association methods, including ReID for identity consistency.
- Used in `Botsort_OSnet.py` and `Botsort_RESnet.py` with OSNet (`osnet_x1_0`) and ResNet50 feature extractors, respectively.

**Implementation**:

- Cloned the BoT-SORT repository and installed dependencies:

  ```bash
  !git clone https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git
  %cd Yolov5_StrongSORT_OSNet
  !pip install -q opencv-python-headless torch torchvision torchaudio ultralytics lap filterpy scipy scikit-learn gdown onnx onnxruntime timm pandas tqdm seaborn
  ```
- Downloaded OSNet weights:

  ```bash
  !gdown https://drive.google.com/uc?id=1h6FGn6v_2PFCV9xRR_9N1a-UXE3L0aD2 -O osnet_x0_25_msmt17.pth
  !mkdir -p osnet_models
  !mv osnet_x0_25_msmt17.pth osnet_models/
  ```
- Processed the video using YOLOv11 and BoT-SORT, with ReID embeddings stored for up to 10 frames.

**Outcome**:

- Successfully tracked players, referees, and the ball with bounding boxes and consistent track IDs.
- OSNet (`Botsort_OSnet.py`) provided better performance than ResNet50 (`Botsort_RESnet.py`) in terms of visual quality of tracking.
- Accuracy was moderate, with occasional ID switches during occlusions or crowded scenes.
- Output videos showed annotated bounding boxes with track IDs, but the accuracy was slightly lower than desired, prompting exploration of other algorithms.

### 2. ByteTrack

**Description**:

- ByteTrack is a high-performance tracking algorithm that associates detections using motion and appearance cues, optimized for speed and accuracy.

**Implementation**:

- Integrated ByteTrack with YOLOv11 in a similar setup to BoT-SORT, using the same input video and model weights.
- Configured with `botsort.yaml` (noting that ByteTrack settings were adapted within the BoT-SORT framework).

**Outcome**:

- Performed well in tracking objects but was less robust than BoT-SORT in maintaining consistent identities during occlusions.
- Bounding boxes were accurate for isolated objects, but ReID performance was weaker compared to BoT-SORT with OSNet.
- Determined that BoT-SORT outperformed ByteTrack due to better handling of crowded scenes and ReID integration.

### 3. Deep SORT

**Description**:

- Deep SORT extends SORT with deep learning-based ReID, using a pretrained ReID model (`mars-small128.pb`) for feature extraction.
- Researched as potentially superior to BoT-SORT and ByteTrack due to its robust ReID capabilities (references: arXiv:2202.13514, Deep SORT GitHub).

**Implementation**:

- Cloned YOLOv5 and Deep SORT repositories:

  ```bash
  !git clone https://github.com/ultralytics/yolov5.git
  %cd yolov5
  !pip install -r requirements.txt
  !git clone https://github.com/nwojke/deep_sort.git
  ```
- Downloaded the Deep SORT ReID model:

  ```bash
  !wget https://github.com/nwojke/deep_sort/raw/master/trained_model/mars-small128.pb -O /content/yolov5/deep_sort/mars-small128.pb
  ```
- Installed additional dependencies:

  ```bash
  !pip install lap scikit-image filterpy
  ```
- Added Deep SORT paths:

  ```python
  import sys
  sys.path.append('/content/yolov5/deep_sort')
  sys.path.append('/content/yolov5/deep_sort/tools')
  ```

**Outcome**:

- Faced significant challenges during setup (see Challenges section).
- Output videos showed oversized bounding boxes that collapsed or overlapped when players gathered, indicating poor handling of crowded scenes.
- Tracking accuracy was lower than BoT-SORT, with frequent ID switches.
- Concluded that Deep SORT underperformed compared to BoT-SORT for this task, despite its theoretical advantages.

### 4. StrongSORT

**Description**:

- StrongSORT builds on SORT with enhanced ReID and motion modeling, designed for robust multi-object tracking (references: AI Scholar StrongSORT, Labellerr Blog).
- Expected to offer high accuracy due to advanced ReID integration.

**Implementation**:

- Cloned the StrongSORT repository:

  ```bash
  !git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet.git
  %cd Yolov7_StrongSORT_OSNet
  !pip install -r requirements.txt
  ```
- Attempted to run tracking with a sample command:

  ```bash
  !python track.py --source 0 --yolo-weights yolov7.pt --classes 16 17  # tracks cats and dogs, only
  ```
- Adapted for the project video (`15sec_input_720p.mp4`) and YOLOv11 weights (`best.pt`).

**Outcome**:

- The code was straightforward to set up, requiring only the repository, video source, and model weights.
- However, execution was extremely slow, likely due to unoptimized configurations or complex ReID computations.
- Faced issues with file corrections in the cloned repository, which hindered smooth execution.
- Did not achieve a complete, high-accuracy output video due to these challenges.

### 5. PRTreID, OC-SORT (Researched, Not Implemented)

- **PRTreID** (arXiv:2401.09942, PRTreID GitHub):
  - Designed for part-based role classification, team affiliation, and person ReID, particularly suited for sports scenarios.
  - Researched as a potential solution but not implemented due to time constraints and complexity of integrating with YOLOv11.
- **OC-SORT** (arXiv:2206.14651):
  - Observation-Centric SORT, focusing on robust tracking under occlusions.
  - Considered but not tested due to prioritization of BoT-SORT and StrongSORT.

## Challenges Encountered

Several challenges arose during the experiments, impacting the ability to achieve optimal tracking and ReID performance:

1. **Deep SORT Installation Issues**:

   - Installing the correct version of Deep SORT in Google Colab was problematic.
   - Frequent session restarts in Colab required reinstalling dependencies, disrupting workflow.
   - Example: After cloning the Deep SORT repository and downloading `mars-small128.pb`, compatibility issues with dependencies caused errors.

2. **Deep SORT Performance**:

   - Output videos showed oversized bounding boxes that collapsed in crowded scenes (e.g., when players gathered).
   - ReID performance was weaker than expected, with frequent ID switches compared to BoT-SORT.

3. **StrongSORT Execution Speed**:

   - Execution of StrongSORT was extremely slow, likely due to unoptimized code or high computational demands of ReID.
   - Example command:

     ```bash
     !python track.py --source 15sec_input_720p.mp4 --y Knapp
     ```
   - Slow processing made it impractical for iterative testing.

4. **StrongSORT File Corrections**:

   - The cloned StrongSORT repository required manual corrections to files (e.g., configuration or path issues), which was time-consuming and error-prone.
   - Specific errors were not detailed, but they prevented smooth execution.

5. **Accuracy Limitations**:

   - BoT-SORT provided moderate accuracy but struggled with occlusions and crowded scenes.
   - Deep SORT and StrongSORT did not improve accuracy as expected, with Deep SORT performing worse than BoT-SORT.

6. **Resource Constraints**:

   - Limited time and computational resources in Colab restricted the ability to fully optimize StrongSORT or test PRTreID and OC-SORT.
   - GPU availability in Colab was helpful but session timeouts disrupted long-running experiments.

## Incomplete Aspects and Next Steps

The project remains incomplete in achieving a highly accurate tracking and ReID output video. The following aspects require further work:

1. **Optimizing StrongSORT**:

   - **Remaining Task**: Address the slow execution and file correction issues in StrongSORT.
   - **Next Steps**:
     - Debug and fix repository file issues by reviewing logs and documentation (Yolov7_StrongSORT_OSNet GitHub).
     - Optimize ReID computations, possibly by reducing the number of frames for embedding history or using a lighter model (e.g., `osnet_x0_25` instead of `osnet_x1_0`).
     - Test on a local machine with higher computational resources to avoid Colab session limits.

2. **Implementing PRTreID**:

   - **Remaining Task**: Integrate PRTreID for sports-specific ReID, leveraging its part-based role classification and team affiliation features.
   - **Next Steps**:
     - Clone the PRTreID repository and follow setup instructions (PRTreID GitHub).
     - Adapt PRTreID to work with YOLOv11 detections, ensuring compatibility with the input video format.
     - Test on a subset of frames to evaluate performance in crowded sports scenes.

3. **Testing OC-SORT**:

   - **Remaining Task**: Implement OC-SORT to compare its occlusion handling with BoT-SORT and StrongSORT.
   - **Next Steps**:
     - Clone the OC-SORT repository and configure it with YOLOv11 (arXiv:2206.14651).
     - Evaluate its performance on the same video dataset, focusing on occlusion scenarios.

4. **Improving Accuracy**:

   - **Remaining Task**: Enhance tracking accuracy to minimize ID switches and handle crowded scenes better.
   - **Next Steps**:
     - Fine-tune YOLOv11 model weights (`best.pt`, `best (2).pt`) on a sports-specific dataset to improve detection accuracy.
     - Adjust ReID similarity threshold (currently 0.7) and embedding history length (currently 10 frames) to optimize identity matching.
     - Experiment with ensemble methods, combining strengths of BoT-SORT and StrongSORT.

5. **Resource Enhancement**:

   - **Remaining Task**: Overcome Colab’s session limits and computational constraints.
   - **Next Steps**:
     - Use a local machine or cloud service with persistent GPU access (e.g., AWS, Google Cloud).
     - Allocate more time to iterate on configurations and test multiple algorithms.

With additional time and resources, I would prioritize optimizing StrongSORT for speed and accuracy, implementing PRTreID for its sports-specific ReID capabilities, and testing OC-SORT to handle occlusions. Fine-tuning the YOLOv11 model and experimenting with ReID parameters would further improve the output video’s quality, ensuring robust tracking and minimal ID switches in crowded scenes.

## References

- BoT-SORT: arXiv:2206.14651
- PRTreID: arXiv:2401.09942, GitHub
- Deep SORT: arXiv:2202.13514, GitHub
- StrongSORT: AI Scholar, Labellerr Blog
- StrongSORT Repository: GitHub