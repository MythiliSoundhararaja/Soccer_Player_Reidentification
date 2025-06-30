# Soccer Player Reidentification

![WhatsApp Image 2025-06-30 at 18 51 50_28905d08](https://github.com/user-attachments/assets/db7b5f79-51c8-4060-b728-fcf8631264ce)


This project contains two Python scripts (`Botsort_OSnet.py` and `Botsort_RESnet.py`) for object tracking in videos using the YOLOv11 model with the BoT-SORT tracker, enhanced with Re-Identification (ReID) using either OSNet or ResNet feature extractors. The scripts process a video, track objects (e.g., players), annotate bounding boxes with track IDs, and save the output as a video file. Additionally, setup instructions for Deep SORT are included, though not used by the provided scripts.

## Prerequisites

### Software and Environment
- **Google Colab** (recommended): The scripts are designed for Google Colab with GPU support for faster processing.
- **Local Environment** (optional): You can run the scripts locally with Python 3.8+ and a GPU (NVIDIA recommended).
- **ffmpeg**: Required for converting output AVI files to MP4 (pre-installed in Colab, must be installed locally).
- **Input Video**: A video file named `15sec_input_720p.mp4` is required.
- **YOLOv11 Model**: Pretrained model weights (`best.pt` for OSNet, `best (2).pt` for ResNet).
- **Git**: Required for cloning repositories.
- **Internet Access**: For downloading model weights and repositories.

### Hardware
- GPU (optional but recommended for faster processing).
- At least 8GB RAM for video processing.

## Dependencies

The required Python packages are listed in `requirements.txt`. Key dependencies include:
- `opencv-python-headless`: For video and image processing.
- `torch`, `torchvision`, `torchaudio`: For PyTorch and feature extraction.
- `ultralytics`: For YOLOv11 model and BoT-SORT tracker.
- `torchreid`: For ReID with OSNet or ResNet (installed via repository).
- `lap`, `filterpy`, `scipy`, `scikit-learn`, `scikit-image`: For tracking and similarity computations.
- `gdown`, `onnx`, `onnxruntime`, `timm`, `pandas`, `tqdm`, `seaborn`: For model downloads and utilities.
- `cython`: For fast metrics computation in TorchReID.
- `numpy==1.26.4`: Specific version for compatibility.

## Setup Instructions

### 1. Google Colab Setup
1. **Open Google Colab**:
   - Create a new notebook or upload the scripts to Colab.
2. **Enable GPU**:
   - Go to `Runtime` > `Change runtime type` > Select `GPU` as the hardware accelerator.
3. **Install ffmpeg**:
   - Run the following command in a Colab cell:
     ```bash
     !apt-get update && apt-get install -y ffmpeg
     ```
4. **Clone BoT-SORT + OSNet Repository**:
   - Run the following commands to clone the repository and install dependencies:
     ```bash
     !git clone https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git
     %cd Yolov5_StrongSORT_OSNet
     !pip install -q opencv-python-headless torch torchvision torchaudio ultralytics lap filterpy scipy scikit-learn gdown onnx onnxruntime timm pandas tqdm seaborn
     ```
5. **Download OSNet Model Weights**:
   - Download and organize the OSNet weights:
     ```bash
     !gdown https://drive.google.com/uc?id=1h6FGn6v_2PFCV9xRR_9N1a-UXE3L0aD2 -O osnet_x0_25_msmt17.pth
     !mkdir -p osnet_models
     !mv osnet_x0_25_msmt17.pth osnet_models/
     ```
6. **Install TorchReID**:
   - Install Cython and clone the TorchReID repository:
     ```bash
     !pip install cython
     !git clone https://github.com/KaiyangZhou/deep-person-reid.git
     %cd deep-person-reid
     ```
   - Install TorchReID dependencies (skipping TensorFlow/JAX):
     ```bash
     !pip install -r requirements.txt --no-deps
     ```
   - Downgrade NumPy for compatibility:
     ```bash
     !pip install numpy==1.26.4
     ```
   - Remove TensorBoard logging:
     ```bash
     !sed -i "s/from torch.utils.tensorboard import SummaryWriter/# removed/g" torchreid/engine/engine.py
     ```
   - Build and install TorchReID:
     ```bash
     !pip install .
     %cd ..
     ```
7. **Install Deep SORT**:
   - Clone YOLOv5 and Deep SORT repositories:
     ```bash
     !git clone https://github.com/ultralytics/yolov5.git
     %cd yolov5
     !pip install -r requirements.txt
     !git clone https://github.com/nwojke/deep_sort.git
     ```
   - Download Deep SORT ReID model:
     ```bash
     !wget https://github.com/nwojke/deep_sort/raw/master/trained_model/mars-small128.pb -O /content/yolov5/deep_sort/mars-small128.pb
     ```
   - Install additional Deep SORT dependencies:
     ```bash
     !pip install lap scikit-image filterpy
     ```
   - Note: The second `wget` command in the provided instructions appears redundant (same file, different URL). Only the first is needed.
8. **Upload Input Files**:
   - Upload the input video (`15sec_input_720p.mp4`) to `/content/`:
     ```python
     from google.colab import files
     files.upload()
     ```
   - Upload the YOLOv11 model weights (`best.pt` for `Botsort_OSnet.py`, `best (2).pt` for `Botsort_RESnet.py`) to `/content/`.
9. **Upload Scripts**:
   - Upload `Botsort_OSnet.py` and `Botsort_RESnet.py` to `/content/`.

### 2. Local Setup (Optional)
1. **Install Python**:
   - Ensure Python 3.8+ is installed.
2. **Install ffmpeg**:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.
3. **Install Git**:
   - Ensure Git is installed for cloning repositories.
4. **Clone and Install BoT-SORT + OSNet**:
   - Clone the repository:
     ```bash
     git clone https://github.com/mikel-brostromself/Yolov5_StrongSORT_OSNet.git
     cd Yolov5_StrongSORT_OSNet
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     pip install opencv-python-headless torch torchvision torchaudio ultralytics lap filterpy scipy scikit-learn gdown onnx onnxruntime timm pandas tqdm seaborn
     ```
   - Download OSNet weights:
     ```bash
     gdown https://drive.google.com/uc?id=1h6FGn6v_2PFCV9xRR_9N1a-UXE3L0aD2 -O osnet_x0_25_msmt17.pth
     mkdir -p osnet_models
     mv osnet_x0_25_msmt17.pth osnet_models/
     ```
5. **Install TorchReID**:
   - Install Cython:
     ```bash
     pip install cython
     ```
   - Clone and install TorchReID:
     ```bash
     git clone https://github.com/KaiyangZhou/deep-person-reid.git
     cd deep-person-reid
     pip install -r requirements.txt --no-deps
     pip install numpy==1.26.4
     sed -i "s/from torch.utils.tensorboard import SummaryWriter/# removed/g" torchreid/engine/engine.py
     pip install .
     cd ..
     ```
6. **Install Deep SORT**:
   - Clone YOLOv5 and Deep SORT:
     ```bash
     git clone https://github.com/ultralytics/yolov5.git
     cd yolov5
     pip install -r requirements.txt
     git clone https://github.com/nwojke/deep_sort.git
     ```
   - Download Deep SORT ReID model:
     ```bash
     wget https://github.com/nwojke/deep_sort/raw/master/trained_model/mars-small128.pb -O deep_sort/mars-small128.pb
     ```
   - Install additional dependencies:
     ```bash
     pip install lap scikit-image filterpy
     ```
7. **Modify Scripts for Local Execution**:
   - Replace `cv2_imshow(frame)` with:
     ```python
     cv2.imshow('Frame', frame)
     cv2.waitKey(1)
     ```
   - Remove Colab-specific code (e.g., `files.download(mp4_path)`, HTML display).
   - Add `cv2.destroyAllWindows()` after `cap.release()`.
   - Add Deep SORT system paths if needed:
     ```python
     import sys
     sys.path.append('yolov5/deep_sort')
     sys.path.append('yolov5/deep_sort/tools')
     ```
8. **Prepare Input Files**:
   - Place `15sec_input_720p.mp4` and model weights (`best.pt`, `best (2).pt`) in the same directory as the scripts.

## Running the Scripts

### In Google Colab
1. **Run `Botsort_OSnet.py`**:
   - From `/content/`, execute:
     ```bash
     !python Botsort_OSnet.py
     ```
   - The script will:
     - Process the video using YOLOv11 with BoT-SORT and OSNet for ReID.
     - Display a sample frame (frame 2).
     - Save the output as `/content/output_with_osnet3.mp4`.
     - Automatically download the output MP4 file.
2. **Run `Botsort_RESnet.py`**:
   - Execute:
     ```bash
     !python Botsort_RESnet.py
     ```
   - The script uses ResNet50 for ReID and saves the output as `/content/output_with_resnet.mp4`.
3. **Deep SORT**:
   - The provided scripts do not use Deep SORT, but the setup is included for potential future use. Refer to the Deep SORT repository for implementation details.

### Locally
1. **Run the Scripts**:
   - From the directory containing the scripts:
     ```bash
     python Botsort_OSnet.py
     ```
     ```bash
     python Botsort_RESnet.py
     ```
   - Outputs will be saved as `output_with_osnet3.mp4` or `output_with_resnet.mp4`.
2. **Deep SORT**:
   - Implement Deep SORT separately using the `yolov5/deep_sort` repository if needed.

## Output
- The scripts generate:
  - An AVI file (`output_with_osnet3.avi` or `output_with_resnet.avi`).
  - An MP4 file (`output_with_osnet3.mp4` or `output_with_resnet.mp4`) with annotated bounding boxes and track IDs.
- In Colab, the MP4 is displayed inline and downloaded automatically.
- Locally, use a video player to view the output MP4 files.

## Notes
- **Model Weights**: Ensure `best.pt` (for OSNet) and `best (2).pt` (for ResNet) are in `/content/` (Colab) or the script directory (local).
- **OSNet Weights**: The scripts use `osnet_x1_0` (auto-downloaded by `torchreid`), but the provided `osnet_x0_25_msmt17.pth` is downloaded for BoT-SORT compatibility.
- **Performance**: GPU acceleration improves processing speed.
- **ReID**: Uses cosine similarity (threshold 0.7) with a 10-frame embedding history.
- **Tracker**: BoT-SORT is configured via `botsort.yaml` (ensure available or use default settings).
- **Empty Files**: `deepsort_try1.py`, `deepsort_try2.py`, and `deepsort_try3.py` are empty and not used.
- **Deep SORT**: Setup is included but not used by the scripts. Refer to `yolov5/deep_sort` for implementation.

## Troubleshooting
- **FileNotFoundError**: Ensure video and model files are in the correct directory.
- **ModuleNotFoundError**: Verify all dependencies and repositories are installed.
- **GPU Issues**: Check `torch.cuda.is_available()`. Scripts fall back to CPU if CUDA is unavailable.
- **ffmpeg Errors**: Ensure `ffmpeg` is installed and in the system PATH.
- **Path Issues**: Ensure `sys.path` includes Deep SORT directories if implementing Deep SORT.

For further assistance, consult:
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
- [Deep SORT](https://github.com/nwojke/deep_sort)
- [Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)
- [OpenCV](https://docs.opencv.org/)
