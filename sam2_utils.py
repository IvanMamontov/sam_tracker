import sys
import cv2
import torch
import os
from pathlib import Path
import numpy as np

# Set CUDA memory management environment variables for better memory handling
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

def clear_gpu_memory():
    """Clear GPU memory cache to free up space"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Clear MPS cache
        torch.mps.empty_cache()
        
def set_memory_efficient_mode():
    """Configure PyTorch for memory efficient operation"""
    if torch.cuda.is_available():
        # Enable memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory
        
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS-specific optimizations
        torch.backends.mps.empty_cache = True


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_sam2_local(predictor_type="image", model_size="base", mask_threshold=0.0, max_hole_area=0, max_sprinkle_area=0):
    """Load SAM2 model locally with proper error handling and quality parameters
    
    Args:
        predictor_type: "image" for image predictor, "video" for video predictor
        model_size: Model size for quality/speed tradeoff:
                   - "tiny": Fastest, lowest quality
                   - "small": Good balance  
                   - "base": Better quality
                   - "large": Best quality (default)
        mask_threshold: Threshold for converting mask logits to binary (0.0 = softer, 1.0 = harder)
        max_hole_area: Fill holes smaller than this area (0 = no filling)
        max_sprinkle_area: Remove disconnected regions smaller than this area (0 = no removal)
    """
    # Point to the sam2_repo location
    root = Path(__file__).parent / "sam2_repo"
    sam2_path = root / "sam2"
    
    # Add sam2 directory to Python path
    if str(sam2_path) not in sys.path:
        sys.path.insert(0, str(sam2_path))
    
    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(f"Could not import SAM2. Make sure it's properly installed in {sam2_path}. Error: {e}")
    
    # Map model size to checkpoint filename
    size_to_checkpoint = {
        "tiny": "sam2.1_hiera_tiny.pt",
        "small": "sam2.1_hiera_small.pt", 
        "base": "sam2.1_hiera_base_plus.pt",
        "large": "sam2.1_hiera_large.pt"
    }
    checkpoint_filename = size_to_checkpoint.get(model_size, "sam2.1_hiera_large.pt")
    checkpoint_path = root / "checkpoints" / checkpoint_filename
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}. Run setup_sam2.sh to download all models.")
    
    # Check if checkpoint is actually a model file (not an error message)
    if checkpoint_path.stat().st_size < 1000:  # Real model should be much larger
        raise ValueError(f"Checkpoint file appears to be invalid (too small: {checkpoint_path.stat().st_size} bytes). "
                        "Please run setup_sam2.sh to download the actual model checkpoints.")
    
    device = _select_device()
    
    # Clear GPU memory and set memory efficient mode before loading model
    clear_gpu_memory()
    set_memory_efficient_mode()
    
    # Map model size to config name
    size_to_config = {
        "tiny": "sam2_hiera_t",
        "small": "sam2_hiera_s", 
        "base": "sam2_hiera_b+",
        "large": "sam2_hiera_l"
    }
    config_name = size_to_config.get(model_size, "sam2_hiera_l")
    
    try:
        if predictor_type == "video":
            # Patch the _load_checkpoint function to use strict=False
            import sam2.build_sam as build_sam_module
            
            # Save original function
            original_load_checkpoint = build_sam_module._load_checkpoint
            
            def patched_load_checkpoint(model, ckpt_path):
                """Load checkpoint with strict=False to handle version mismatches"""
                if ckpt_path is not None:
                    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
                    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
                    if unexpected_keys:
                        print(f"Warning: Ignoring unexpected keys in video checkpoint: {unexpected_keys[:3]}...")
            
            try:
                # Temporarily replace the function
                build_sam_module._load_checkpoint = patched_load_checkpoint
                
                # Now build video predictor with correct config
                config_file = f"{config_name}.yaml"
                predictor = build_sam2_video_predictor(
                    config_file=config_file,
                    ckpt_path=str(checkpoint_path),
                    device=device,
                    vos_optimized=False
                )
                return predictor
            finally:
                # Restore original function
                build_sam_module._load_checkpoint = original_load_checkpoint
        else:
            # Respect requested model size for config and checkpoint
            size_to_config_file = {
                "tiny": "sam2_hiera_t.yaml",
                "small": "sam2_hiera_s.yaml",
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
            }
            config_file = size_to_config_file.get(model_size, "sam2_hiera_l.yaml")

            # Prefer SAM 2.1 checkpoints if present (they store weights under the 'model' key)
            checkpoint_filename = size_to_checkpoint.get(model_size, "sam2.1_hiera_large.pt")
            checkpoint_path = root / "checkpoints" / checkpoint_filename

            # Fallback to older naming if the above doesn't exist
            if not checkpoint_path.exists():
                fallback_name = checkpoint_filename.replace("sam2.1_", "sam2_")
                checkpoint_path = root / "checkpoints" / fallback_name

            # Build model from config, then load weights manually with strict=False
            sam2_model = build_sam2(config_file, ckpt_path=None, device=device)

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            missing_keys, unexpected_keys = sam2_model.load_state_dict(state_dict, strict=False)
            if unexpected_keys:
                print(
                    f"Warning: Unexpected keys in checkpoint (probably SAM2.1 vs SAM2.0): {str(unexpected_keys)[:200]}..."
                )
            if missing_keys:
                print(
                    f"Warning: Missing keys when loading checkpoint: {str(missing_keys)[:200]}..."
                )

            predictor = SAM2ImagePredictor(
                sam2_model,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area,
            )
            return predictor
    except Exception as e:
        raise RuntimeError(f"Failed to build SAM2 predictor: {e}")

def _prep_mask_for_frame(mask_in, frame_shape):
    """
    Returns (mask_bool_for_logic, mask_f01_for_save @ frame size)
    """
    h, w = frame_shape[:2]
    if mask_in is None:
        return None, None

    m = mask_in
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]

    # to float 0..1 for robust resize
    if m.dtype == np.bool_:
        mf = m.astype(np.float32)
        interp = cv2.INTER_NEAREST
    elif m.dtype == np.uint8:
        mf = (m.astype(np.float32) / 255.0)
        interp = cv2.INTER_LINEAR
    else:
        mf = m.astype(np.float32)
        interp = cv2.INTER_LINEAR

    if mf.shape != (h, w):
        mf = cv2.resize(mf, (w, h), interpolation=interp)

    m_bool = mf > 0.5
    return m_bool, np.clip(mf, 0.0, 1.0)

def _clean_binary(m_bool: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Remove tiny FG islands and fill tiny enclosed holes.
    Thresholds are fractions of the frame area (resolution-agnostic).
    """
    # Tune here if needed:
    MIN_ISLAND_FRAC = 5e-5   # ~46 px at 1280x720
    MAX_HOLE_FRAC   = 1e-3   # ~920 px at 1280x720

    min_island = int(MIN_ISLAND_FRAC * H * W)
    max_hole   = int(MAX_HOLE_FRAC   * H * W)

    m = m_bool.astype(np.uint8)

    # 1) remove tiny FG islands
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(num, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= max(1, min_island):
            keep[i] = 1
    m = keep[lab]

    # 2) fill small holes (BG components fully inside the image)
    inv = (1 - m).astype(np.uint8)
    num2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num2):
        x, y, w, h, area = stats2[i, :5]
        touches_border = (x == 0 or y == 0 or (x + w) >= W or (y + h) >= H)
        if (not touches_border) and area <= max_hole:
            m[lab2 == i] = 1

    # small morphological close to seal pinholes along edges
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
    return m.astype(bool)

def save_masks_and_frames(video_path, masks, output_dir="output/sam2_tracking",
                          use_exr=False, union=None, occlusions=None, save_frames=True,
                          refine_edges=False, choke_px=2, feather_px=4, guided=False, frame_offset=0):
    """
    If refine_edges=True, masks are cleaned (remove tiny islands, fill tiny holes),
    then choked + feathered (optionally guided) *before writing*.
    All logic (union/occlusions/overlay) still uses boolean masks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_dir = output_path / "frames" if save_frames else None
    masks_dir = output_path / "masks"
    overlay_dir = output_path / "overlays"
    if save_frames:
        frames_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    # occluders
    occluders_dir = None
    if occlusions is not None:
        occluders_dir = output_path / "occluders"
        occluders_dir.mkdir(exist_ok=True)

    # ---------- local helpers ----------
    def _refine_alpha(mask_bool: np.ndarray, plate_bgr: np.ndarray) -> np.ndarray:
        """0/1 -> soft 0..1 via choke + feather (+ optional guided)."""
        a = mask_bool.astype(np.uint8)
        if choke_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(choke_px)+1, 2*int(choke_px)+1))
            a = cv2.erode(a, k, 1)
        soft = cv2.GaussianBlur(a.astype(np.float32), (0, 0),
                                sigmaX=max(1, int(feather_px)), sigmaY=max(1, int(feather_px)))
        soft = np.clip(soft, 0.0, 1.0)
        if guided and hasattr(cv2, "ximgproc"):
            guide = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            soft = cv2.ximgproc.guidedFilter(guide, soft, radius=max(1, int(feather_px)), eps=1e-4)
        return soft
    # -----------------------------------

    src = Path(video_path)
    frame_count = 0

    # ----- sequence -----
    if src.is_dir():
        jpgs = sorted([p for p in src.iterdir() if p.suffix.lower() == ".jpg"])
        if not jpgs:
            jpgs = sorted([p for p in src.iterdir()
                           if p.suffix.lower() == ".jpg" and p.name.startswith("frame_")])

        for p in jpgs:
            frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {p}")

            frame_ext = "exr" if use_exr else "png"
            mask_ext  = "exr" if use_exr else "png"
            overlay_ext = "png"

            # Use frame_offset for file naming
            actual_frame_num = frame_count + frame_offset
            
            if save_frames:
                fn = f"frame_{actual_frame_num:06d}.{frame_ext}"
                if use_exr:
                    cv2.imwrite(str(frames_dir / fn),
                                frame.astype('float32') / 255.0,
                                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                else:
                    cv2.imwrite(str(frames_dir / fn), frame)

            if frame_count < len(masks):
                mask_in = masks[frame_count]
                if mask_in is not None:
                    m_bool, m_f01 = _prep_mask_for_frame(mask_in, frame.shape)

                    # clean pinholes/specks (fixes the type of holes you showed)
                    if m_bool is not None and refine_edges:
                        m_bool = _clean_binary(m_bool, *frame.shape[:2])

                    # what we write to disk
                    if refine_edges and m_bool is not None:
                        m_save = _refine_alpha(m_bool, frame)  # float 0..1
                    else:
                        m_save = m_f01  # preserve incoming grayscale if any

                    fnm = f"mask_{actual_frame_num:06d}.{mask_ext}"
                    if use_exr:
                        cv2.imwrite(str(masks_dir / fnm),
                                    m_save.astype('float32'),
                                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                    else:
                        cv2.imwrite(str(masks_dir / fnm),
                                    (np.clip(m_save, 0, 1) * 255).astype('uint8'))

                    # overlay uses boolean (cleaned) for QC
                    if m_bool is not None:
                        ov = frame.copy()
                        ov[m_bool] = [0, 0, 255]
                        vis = cv2.addWeighted(frame, 0.7, ov, 0.3, 0)
                        cv2.imwrite(str(overlay_dir / f"overlay_{actual_frame_num:06d}.{overlay_ext}"), vis)

            frame_count += 1

    # ----- video -----
    else:
        cap = cv2.VideoCapture(str(src))
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_ext = "exr" if use_exr else "png"
            mask_ext  = "exr" if use_exr else "png"
            overlay_ext = "png"
            
            # Use frame_offset for file naming
            actual_frame_num = frame_count + frame_offset

            if save_frames:
                fn = f"frame_{actual_frame_num:06d}.{frame_ext}"
                if use_exr:
                    cv2.imwrite(str(frames_dir / fn),
                                frame.astype('float32') / 255.0,
                                [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                else:
                    cv2.imwrite(str(frames_dir / fn), frame)

            if frame_count < len(masks):
                mask_in = masks[frame_count]
                if mask_in is not None:
                    m_bool, m_f01 = _prep_mask_for_frame(mask_in, frame.shape)

                    if m_bool is not None and refine_edges:
                        m_bool = _clean_binary(m_bool, *frame.shape[:2])

                    if refine_edges and m_bool is not None:
                        m_save = _refine_alpha(m_bool, frame)
                    else:
                        m_save = m_f01

                    fnm = f"mask_{actual_frame_num:06d}.{mask_ext}"
                    if use_exr:
                        cv2.imwrite(str(masks_dir / fnm),
                                    m_save.astype('float32'),
                                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
                    else:
                        cv2.imwrite(str(masks_dir / fnm),
                                    (np.clip(m_save, 0, 1) * 255).astype('uint8'))

                    if m_bool is not None:
                        ov = frame.copy()
                        ov[m_bool] = [0, 0, 255]
                        vis = cv2.addWeighted(frame, 0.7, ov, 0.3, 0)
                        cv2.imwrite(str(overlay_dir / f"overlay_{actual_frame_num:06d}.{overlay_ext}"), vis)

            frame_count += 1
        cap.release()

    # union
    if union is not None:
        union_filename = output_path / "union.png"
        u = union
        if len(u.shape) == 3 and u.shape[0] == 1:
            u = u.squeeze(0)
        u_bool = u > 0.5 if u.dtype in (np.float32, np.float64) else u.astype(bool)
        cv2.imwrite(str(union_filename), (u_bool * 255).astype('uint8'))
        print(f"✅ Saved union mask to {union_filename}")

    # occlusions
    if occlusions is not None and occluders_dir is not None:
        saved = 0
        for i, occ in enumerate(occlusions):
            if occ is None:
                continue
            if len(occ.shape) == 3 and occ.shape[0] == 1:
                occ = occ.squeeze(0)
            occ_bool = occ > 0.5 if occ.dtype in (np.float32, np.float64) else occ.astype(bool)
            cv2.imwrite(str(occluders_dir / f"occ_{i:04d}.png"), (occ_bool * 255).astype('uint8'))
            saved += 1
        print(f"✅ Saved {saved} occlusion masks to {occluders_dir}")

    fmt = "EXR (half-float)" if use_exr else "PNG"
    if save_frames:
        print(f"✅ Saved {frame_count} frames and {len(masks)} masks to {output_path} [{fmt}]")
        print(f"   - Original frames: {frames_dir}")
    else:
        print(f"✅ Saved {len(masks)} masks to {output_path} [{fmt}]")
    print(f"   - Masks: {masks_dir}")
    print(f"   - Overlays: {overlay_dir} [PNG]")
    if union is not None:
        print(f"   - Union mask: {output_path / 'union.png'}")
    if occlusions is not None:
        print(f"   - Occlusions: {output_path / 'occluders'} [PNG]")
    if refine_edges:
        mode = "guided" if (guided and hasattr(cv2, "ximgproc")) else "gaussian"
        print(f"   - Refine edges: ON (choke={choke_px}, feather={feather_px}, mode={mode}; "
              f"hole-fill + despeckle enabled)")
    return output_path


def predict_with_quality_params(predictor, image, point_coords, point_labels=None, 
                               multimask_output=True, return_logits=False, 
                               pred_iou_thresh=0.8, stability_score_thresh=0.95):
    """Enhanced prediction with quality parameters exposed
    
    Args:
        predictor: SAM2 predictor instance
        image: Input image
        point_coords: Point coordinates for prompts
        point_labels: Point labels (1=foreground, 0=background)
        multimask_output: Whether to generate 3 masks (better for ambiguous prompts)
        return_logits: Return raw logits instead of binary masks
        pred_iou_thresh: Filter masks below this IoU confidence (0.0-1.0)
        stability_score_thresh: Filter unstable masks below this score (0.0-1.0)
    """
    import numpy as np
    
    # Set image if not already set
    if hasattr(predictor, 'set_image'):
        predictor.set_image(image)
    
    # Default labels to foreground if not provided
    if point_labels is None:
        point_labels = np.ones(len(point_coords))
    
    # Run prediction
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask_output,
        return_logits=return_logits
    )
    
    # Apply quality filtering
    if pred_iou_thresh > 0.0:
        good_masks = scores >= pred_iou_thresh
        masks = masks[good_masks]
        scores = scores[good_masks]
        logits = logits[good_masks]
        print(f"Quality filter: {good_masks.sum()}/{len(good_masks)} masks passed IoU threshold {pred_iou_thresh}")
    
    # Calculate stability scores if requested
    if stability_score_thresh > 0.0:
        stability_scores = []
        for mask in masks:
            # Simple stability calculation: how much the mask changes with small threshold shifts
            if return_logits:
                stability = _calculate_stability_score(mask, 0.0, 0.05)
            else:
                stability = 1.0  # Binary masks are already stable
            stability_scores.append(stability)
        
        stable_masks = np.array(stability_scores) >= stability_score_thresh
        masks = masks[stable_masks]
        scores = scores[stable_masks] 
        logits = logits[stable_masks]
        print(f"Stability filter: {stable_masks.sum()}/{len(stable_masks)} masks passed stability threshold {stability_score_thresh}")
    
    return masks, scores, logits


def _calculate_stability_score(mask_logits, mask_threshold, threshold_offset):
    """Calculate how stable a mask is under small threshold changes"""
    import torch
    
    # Convert to tensor if needed
    if not isinstance(mask_logits, torch.Tensor):
        mask_logits = torch.from_numpy(mask_logits)
    
    # Get masks at different thresholds
    mask_1 = mask_logits > (mask_threshold + threshold_offset)
    mask_2 = mask_logits > (mask_threshold - threshold_offset)
    
    # Calculate intersection over union
    intersection = (mask_1 & mask_2).sum()
    union = (mask_1 | mask_2).sum()
    
    if union == 0:
        return 1.0
    return float(intersection / union)

def _sample_points_inside_mask(mask, k=15):
    """Sample points inside mask (adapted from dynamic_cluster_filtering.py)."""
    m = mask.astype(np.uint8)
    if m.sum() == 0 or k <= 0:
        return []

    H, W = m.shape

    # Mild erosion to stay safely inside
    safe_mask = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)

    # If erosion removes everything, use original
    if safe_mask.sum() == 0:
        safe_mask = m

    # Get all valid points
    ys, xs = np.where(safe_mask > 0)
    if len(xs) == 0:
        return []

    points = []

    # Include some edge points (20%)
    k_edge = min(3, k // 5) if k >= 10 else 0

    if k_edge > 0:
        # Find edge pixels
        edge_mask = m & (~safe_mask)
        if edge_mask.sum() > 0:
            ys_e, xs_e = np.where(edge_mask > 0)
            step = max(1, len(xs_e) // k_edge)
            for i in range(0, min(k_edge, len(xs_e)), step):
                points.append((int(xs_e[i]), int(ys_e[i])))

    # Fill remaining with interior points
    k_interior = k - len(points)
    if k_interior > 0:
        n_available = len(xs)
        n_samples = min(k_interior, n_available)

        # Evenly spaced indices for good spatial distribution
        indices = np.linspace(0, n_available - 1, num=n_samples, dtype=int)

        for idx in indices:
            points.append((int(xs[idx]), int(ys[idx])))

    return points[:k]