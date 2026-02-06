import cv2
import numpy as np
import json
import os
import tempfile
import shutil
from PIL import Image

# Global variables
current_frames_dir = None
num_frames = 0
video_segments = {}
first_frame_array = None
polygon_points = []

def extract_frames(video_path, fps=5):
    global current_frames_dir, num_frames, first_frame_array
    
    current_frames_dir = tempfile.mkdtemp()
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    frame_interval = max(1, int(video_fps / fps))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f'{current_frames_dir}/{saved_count:05d}.jpg', frame)
            saved_count += 1
        frame_count += 1
    
    cap.release()
    num_frames = saved_count
    
    if num_frames == 0:
        return None, "Error: Could not extract frames"
    
    first_frame_array = cv2.imread(f'{current_frames_dir}/00000.jpg')
    first_frame_array = cv2.cvtColor(first_frame_array, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(first_frame_array), f"Extracted {num_frames} frames. Now click points around the shark!"

def process_video(video_file, frame_rate):
    global polygon_points
    polygon_points = []
    if video_file is None:
        return None, "Please upload a video"
    return extract_frames(video_file, fps=frame_rate)

def add_point(image, evt):
    global polygon_points, first_frame_array
    if first_frame_array is None:
        return image, "Upload a video first"
    
    x, y = evt.index
    polygon_points.append((x, y))
    
    img = first_frame_array.copy()
    for i, pt in enumerate(polygon_points):
        cv2.circle(img, pt, 6, (255, 0, 0), -1)
        if i > 0:
            cv2.line(img, polygon_points[i-1], pt, (0, 255, 0), 2)
    
    return Image.fromarray(img), f"Points: {len(polygon_points)}"

def close_polygon():
    global polygon_points, first_frame_array
    if len(polygon_points) < 3:
        return Image.fromarray(first_frame_array), "Need at least 3 points"
    
    img = first_frame_array.copy()
    pts = np.array(polygon_points, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return Image.fromarray(img), f"Polygon closed with {len(polygon_points)} points!"

def clear_points():
    global polygon_points, first_frame_array
    polygon_points = []
    if first_frame_array is None:
        return None, "Upload a video first"
    return Image.fromarray(first_frame_array), "Cleared"

def run_sam2():
    global video_segments, polygon_points, current_frames_dir, num_frames, first_frame_array
    
    if len(polygon_points) < 3:
        return "Draw a polygon first", None
    
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        
        if not os.path.exists("checkpoints/sam2_hiera_large.pt"):
            os.makedirs("checkpoints", exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
                "checkpoints/sam2_hiera_large.pt"
            )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = build_sam2_video_predictor("sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt", device=device)
        
        h, w = first_frame_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        
        inference_state = predictor.init_state(video_path=current_frames_dir)
        predictor.add_new_mask(inference_state=inference_state, frame_idx=0, obj_id=1, mask=mask)
        
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        preview = []
        for idx in list(video_segments.keys())[::max(1, len(video_segments)//4)][:4]:
            frame = cv2.imread(f'{current_frames_dir}/{idx:05d}.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if 1 in video_segments[idx]:
                mask = video_segments[idx][1]
                overlay = frame.copy()
                overlay[mask > 0] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            preview.append(Image.fromarray(frame))
        
        return f"Tracked through {len(video_segments)} frames!", preview
    
    except Exception as e:
        return f"Error: {str(e)}", None

def create_comparison_video():
    global video_segments, current_frames_dir, num_frames
    
    if not video_segments:
        return None, "Run SAM 2 first"
    
    first_frame = cv2.imread(f'{current_frames_dir}/00000.jpg')
    h, w = first_frame.shape[:2]
    
    video_path = f"{tempfile.gettempdir()}/comparison_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (w * 2, h))
    
    for frame_idx in sorted(video_segments.keys()):
        frame_path = f'{current_frames_dir}/{frame_idx:05d}.jpg'
        if not os.path.exists(frame_path):
            continue
        
        original = cv2.imread(frame_path)
        cv2.putText(original, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        tracked = original.copy()
        
        if frame_idx in video_segments and 1 in video_segments[frame_idx]:
            mask = video_segments[frame_idx][1]
            overlay = tracked.copy()
            overlay[mask > 0] = [0, 255, 0]
            tracked = cv2.addWeighted(tracked, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tracked, contours, -1, (0, 255, 0), 2)
        
        cv2.putText(tracked, "SAM 2 TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(original, f"Frame: {frame_idx}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(tracked, f"Frame: {frame_idx}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        combined = np.hstack([original, tracked])
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
        out.write(combined)
    
    out.release()
    return video_path, f"Comparison video created with {len(video_segments)} frames!"

def create_overlay_video():
    global video_segments, current_frames_dir
    
    if not video_segments:
        return None, "Run SAM 2 first"
    
    first_frame = cv2.imread(f'{current_frames_dir}/00000.jpg')
    h, w = first_frame.shape[:2]
    
    video_path = f"{tempfile.gettempdir()}/tracking_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (w, h))
    
    for frame_idx in sorted(video_segments.keys()):
        frame_path = f'{current_frames_dir}/{frame_idx:05d}.jpg'
        if not os.path.exists(frame_path):
            continue
        
        frame = cv2.imread(frame_path)
        
        if frame_idx in video_segments and 1 in video_segments[frame_idx]:
            mask = video_segments[frame_idx][1]
            overlay = frame.copy()
            overlay[mask > 0] = [0, 255, 0]
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONTFONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
    
    out.release()
    return video_path, "Overlay video created!"

def check_missing_frames():
    global video_segments, num_frames
    
    if not video_segments:
        return "Run SAM 2 first"
    
    all_frames = set(range(num_frames))
    annotated = set(video_segments.keys())
    missing = sorted(all_frames - annotated)
    
    total = num_frames
    tracked = len(annotated)
    missed = len(missing)
    coverage = (tracked / total) * 100 if total > 0 else 0
    
    report = f"TRACKING REPORT\n\nTotal frames: {total}\nTracked: {tracked}\nMissing: {missed}\nCoverage: {coverage:.1f}%\n\n"
    
    if missing:
        report += f"Missing frames: {missing[:20]}"
        if len(missing) > 20:
            report += f"\n... and {len(missing) - 20} more"
    else:
        report += "No missing frames! SAM 2 tracked everything."
    
    return report

def export_annotations(class_name):
    global video_segments, current_frames_dir
    
    if not video_segments:
        return None, "Run SAM 2 first"
    
    export_dir = tempfile.mkdtemp()
    os.makedirs(f"{export_dir}/images", exist_ok=True)
    
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": class_name}]}
    ann_id = 1
    
    for idx, masks in video_segments.items():
        if 1 not in masks:
            continue
        mask = masks[1]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        polygon = max(contours, key=cv2.contourArea).flatten().tolist()
        if len(polygon) < 6:
            continue
        
        shutil.copy(f'{current_frames_dir}/{idx:05d}.jpg', f'{export_dir}/images/{idx:05d}.jpg')
        h, w = mask.shape
        coco["images"].append({"id": idx, "file_name": f'{idx:05d}.jpg', "width": w, "height": h})
        coco["annotations"].append({"id": ann_id, "image_id": idx, "category_id": 1, "segmentation": [polygon], "iscrowd": 0})
        ann_id += 1
    
    with open(f'{export_dir}/_annotations.coco.json', 'w') as f:
        json.dump(coco, f)
    
    zip_path = f"{tempfile.gettempdir()}/annotations.zip"
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', export_dir)
    
    return zip_path, f"Exported {len(coco['images'])} frames!"
