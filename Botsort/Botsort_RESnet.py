
import torch
from torchreid.utils import FeatureExtractor
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path
import os
import base64
from google.colab.patches import cv2_imshow
from IPython.display import HTML, display
from google.colab import files
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Initialize OSNet extractor with a more robust model
extractor = FeatureExtractor(
    model_name='resnet50',  # Upgraded from osnet_x0_25 for better accuracy
    model_path='',  # auto-download pretrained weights
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ✅ Track history and ReID embedding storage
track_history = defaultdict(list)
player_embeddings = {}  # {id: (track_id, [feature_history])}
current_frame_embeddings = {}  # Temporary storage for current frame
embedding_history = defaultdict(list)  # Store recent embeddings per ID

# ✅ Load YOLOv11 model
model = YOLO("/content/best (2).pt")
names = model.model.names
video_path = "/content/15sec_input_720p.mp4"

# ✅ Check input
if not Path(video_path).exists():
    raise FileNotFoundError(f"Source path '{video_path}' does not exist.")

cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps > 0 else 30

# ✅ Output video setup
output_path = "/content/output_with_resnet.avi"
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'XVID'),
                      fps, (frame_width, frame_height))

# ✅ Process each frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, tracker="botsort.yaml")

    boxes = results[0].boxes.xyxy.cpu().numpy()
    clss = results[0].boxes.cls.cpu().tolist()

    if results[0].boxes.id is None:
        continue

    track_ids = results[0].boxes.id.int().cpu().tolist()
    annotator = Annotator(frame, line_width=2, example=str(names))

    # Clear current frame embeddings
    current_frame_embeddings.clear()

    for box, track_id, cls in zip(boxes, track_ids, clss):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls]} : {track_id}"
        annotator.box_label([x1, y1, x2, y2], label, (218, 100, 255))

        # ✅ Crop and extract embedding for player
        crop = frame[y1:y2, x1:x2]
        if crop.size != 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            features = extractor([crop_rgb])[0].cpu().numpy()
            current_frame_embeddings[track_id] = features
            embedding_history[track_id].append(features)
            if len(embedding_history[track_id]) > 10:  # Limit history to 10 frames
                embedding_history[track_id].pop(0)

        # ✅ Draw center point (yellow lines removed)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        track = track_history[track_id]
        track.append((cx, cy))
        if len(track) > 30:
            track.pop(0)
        cv2.circle(frame, (cx, cy), 5, (235, 219, 11), -1)

    # ✅ Re-identify and update track IDs based on embeddings
    if player_embeddings and current_frame_embeddings:
        for new_id, new_feature in current_frame_embeddings.items():
            max_similarity = -1
            best_match_id = new_id
            for old_id, (old_track_id, old_features) in player_embeddings.items():
                # Average the historical embeddings for better matching
                if old_features:
                    old_feature_avg = np.mean(old_features, axis=0)
                    similarity = cosine_similarity([new_feature], [old_feature_avg])[0][0]
                    if similarity > max_similarity and similarity > 0.7:  # Lowered threshold to 0.7
                        max_similarity = similarity
                        best_match_id = old_track_id
            # Update track history and embeddings
            if best_match_id != new_id:
                track_history[best_match_id] = track_history[new_id]
                track_history[new_id] = []
                player_embeddings[best_match_id] = (best_match_id, embedding_history[best_match_id])
                if new_id in player_embeddings:
                    del player_embeddings[new_id]

    # Update player_embeddings with current frame data
    for track_id, features in current_frame_embeddings.items():
        player_embeddings[track_id] = (track_id, embedding_history[track_id])

    # Show one sample frame
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 2:
        cv2_imshow(frame)

    out.write(frame)

cap.release()
out.release()

# ✅ Convert to mp4 for Colab
mp4_path = "/content/output_with_resnet.mp4"
!ffmpeg -y -i "$output_path" -vcodec libx264 "$mp4_path" -loglevel quiet

# ✅ Display inline
if os.path.exists(mp4_path):
    mp4 = open(mp4_path, 'rb').read()
    data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
    display(HTML(f"""
    <video width=640 controls>
        <source src="{data_url}" type="video/mp4">
    </video>
    """))
    files.download(mp4_path)