import cv2
from pathlib import Path
import numpy as np
import torch
from sam2_utils import save_masks_and_frames, load_sam2_local, clear_gpu_memory


class ROITracker:
    def __init__(self, predictor):
        """
        ROITracker for SAM2 video tracking.
        
        Note: Preprocessing is now handled by pipelines using sam2_preprocessing.py
        before calling track(). This ensures single-pass preprocessing with auto-detection.
        """
        self.predictor = predictor

    def track(self, video_path, prompts, frame_idx=0, save_to=None, use_exr=False, save_occlusions=False,
              save_frames=False, verbose=True, refine_edges=False, gap_fill_threshold=0.1, frame_offset=0):
        """
        Now expects video_path to be a directory with a JPG sequence: 00000.jpg, 00001.jpg, ...
        gap_fill_threshold: minimum mask area ratio to consider object as present
        """
        seq_dir, frame_count = self._validate_image_sequence(video_path)
        if verbose:
            print(f"🎬 Using image sequence: {seq_dir}  frames={frame_count}")

        # Auto-detect if we need chunking based on frame count and available memory
        chunk_size = self._get_optimal_chunk_size(frame_count, verbose)
        
        if frame_count > chunk_size:
            if verbose:
                print(f"📦 Large sequence detected. Using chunked processing: {frame_count} frames → chunks of {chunk_size}")
            return self._track_chunked(video_path, prompts, frame_idx, save_to, use_exr, save_occlusions,
                                     save_frames, verbose, refine_edges, gap_fill_threshold, frame_offset, chunk_size)
        
        # Single chunk processing for smaller sequences
        return self._track_single(video_path, prompts, frame_idx, save_to, use_exr, save_occlusions,
                                save_frames, verbose, refine_edges, gap_fill_threshold, frame_offset)

    def _get_optimal_chunk_size(self, frame_count, verbose=True):
        """Dynamically determine optimal chunk size based on available memory and device"""
        import psutil
        
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Device-specific base chunk sizes (conservative estimates)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS (Apple Silicon) - more conservative due to unified memory
            base_chunk = 300
            memory_factor = min(available_memory_gb / 8.0, 2.0)  # Scale based on 8GB baseline
        elif torch.cuda.is_available():
            # CUDA - can handle larger chunks
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            base_chunk = 500
            memory_factor = min(gpu_memory_gb / 12.0, 3.0)  # Scale based on 12GB baseline
        else:
            # CPU - most conservative
            base_chunk = 200
            memory_factor = min(available_memory_gb / 16.0, 1.5)  # Scale based on 16GB baseline
        
        chunk_size = int(base_chunk * memory_factor)
        chunk_size = max(100, min(chunk_size, 1000))  # Clamp between 100-1000
        
        if verbose:
            device_name = "MPS" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else \
                         "CUDA" if torch.cuda.is_available() else "CPU"
            print(f"🧠 Memory-aware chunking: {device_name} device, {available_memory_gb:.1f}GB available → chunk_size={chunk_size}")
        
        return chunk_size

    def _track_single(self, video_path, prompts, frame_idx=0, save_to=None, use_exr=False, save_occlusions=False,
                     save_frames=False, verbose=True, refine_edges=False, gap_fill_threshold=0.1, frame_offset=0):
        """Original single-pass tracking for smaller sequences"""
        seq_dir, frame_count = self._validate_image_sequence(video_path)
        
        # Clear GPU memory before initializing tracking state
        clear_gpu_memory()

        # init state with the directory of JPGs
        inference_state = self.predictor.init_state(video_path=str(seq_dir))

        # Reset any previous state to ensure clean tracking
        self.predictor.reset_state(inference_state)

        # normalize prompts
        normalized_prompts = self._normalize_prompts(prompts, frame_idx, verbose)

        # validate prompt frame indices
        bad = [p["frame_idx"] for p in normalized_prompts if not (0 <= p["frame_idx"] < frame_count)]
        if bad:
            raise ValueError(f"Prompt frame_idx out of range for sequence with {frame_count} frames: {bad}")

        keyframe = min(p["frame_idx"] for p in normalized_prompts)
        if verbose:
            print(f"🔑 Prompts on frames: {[p['frame_idx'] for p in normalized_prompts]}, keyframe: {keyframe}")

        # add prompts
        self._add_prompts_to_state(inference_state, normalized_prompts, verbose)

        # First pass: standard bidirectional propagation
        all_masks_by_object = self._propagate_bidirectional(inference_state, frame_count, keyframe)

        # Flatten to single-object list
        masks = [None] * frame_count
        if 0 in all_masks_by_object:
            for f_idx, mask in all_masks_by_object[0].items():
                masks[f_idx] = mask

        # Second pass: detect gaps and fill them
        masks = self._fill_occlusion_gaps(
            masks, inference_state, frame_count, gap_fill_threshold, verbose
        )

        # Clear GPU memory after tracking is complete
        clear_gpu_memory()

        if verbose:
            print(f"✅ Tracked object across {sum(1 for m in masks if m is not None)} frames [single pass]")

        # save
        if save_to:
            if not any(mask is not None for mask in masks):
                print("❌ No masks generated")
                return masks, None

            union_mask = self.calculate_union_mask(masks) if save_occlusions else None
            occlusion_masks = self.calculate_occlusions(masks, keyframe) if save_occlusions else None

            out = save_masks_and_frames(
                video_path=str(seq_dir),  # Use original sequence for saving
                masks=masks,
                output_dir=save_to,
                use_exr=use_exr,
                union=union_mask,
                occlusions=occlusion_masks,
                save_frames=save_frames,
                refine_edges=refine_edges,
                frame_offset=frame_offset
            )
            return masks, out

        return masks

    def _track_chunked(self, video_path, prompts, frame_idx=0, save_to=None, use_exr=False, save_occlusions=False,
                      save_frames=False, verbose=True, refine_edges=False, gap_fill_threshold=0.1, frame_offset=0, chunk_size=500):
        """Smart chunked tracking with overlap handling for large sequences"""
        seq_dir, frame_count = self._validate_image_sequence(video_path)
        
        # Normalize prompts once and store for chunk processing
        self.original_prompts = prompts  # Store original prompts for chunk processing
        normalized_prompts = self._normalize_prompts(prompts, frame_idx, verbose)
        keyframe = min(p["frame_idx"] for p in normalized_prompts)
        
        # Initialize final masks array
        final_masks = [None] * frame_count
        
        # Calculate chunks with overlap
        overlap = min(50, chunk_size // 4)  # 25% overlap, max 50 frames
        chunks = self._calculate_chunks(frame_count, chunk_size, overlap, keyframe, normalized_prompts)
        
        if verbose:
            print(f"📦 Processing {len(chunks)} chunks with {overlap}-frame overlap")
        
        # Process each chunk
        for i, (start, end, chunk_keyframe, chunk_prompts) in enumerate(chunks):
            if verbose:
                print(f"🔄 Chunk {i+1}/{len(chunks)}: frames {start}-{end} (keyframe: {chunk_keyframe})")
            
            # Create temporary chunk directory
            chunk_masks = self._process_chunk(seq_dir, start, end, chunk_keyframe, chunk_prompts, verbose)
            
            # Merge chunk results into final masks
            self._merge_chunk_masks(final_masks, chunk_masks, start, end, overlap, verbose)
            
            # Clear memory between chunks
            clear_gpu_memory()
        
        if verbose:
            print(f"✅ Chunked tracking complete: {sum(1 for m in final_masks if m is not None)}/{frame_count} frames")
        
        # Save results
        if save_to:
            if not any(mask is not None for mask in final_masks):
                print("❌ No masks generated")
                return final_masks, None

            union_mask = self.calculate_union_mask(final_masks) if save_occlusions else None
            occlusion_masks = self.calculate_occlusions(final_masks, keyframe) if save_occlusions else None

            out = save_masks_and_frames(
                video_path=str(seq_dir),
                masks=final_masks,
                output_dir=save_to,
                use_exr=use_exr,
                union=union_mask,
                occlusions=occlusion_masks,
                save_frames=save_frames,
                refine_edges=refine_edges,
                frame_offset=frame_offset
            )
            return final_masks, out

        return final_masks

    def _calculate_chunks(self, frame_count, chunk_size, overlap, keyframe, normalized_prompts):
        """Calculate optimal chunk boundaries with keyframe consideration"""
        chunks = []
        
        # Generate all chunks
        for i in range(0, frame_count, chunk_size - overlap):
            start = max(0, i)
            end = min(frame_count - 1, i + chunk_size - 1)
            
            # Determine chunk keyframe and prompts
            if start <= keyframe <= end:
                # This chunk contains the original keyframe
                chunk_keyframe = keyframe - start  # Relative to chunk start
                chunk_prompts = self._adjust_prompts_for_chunk(normalized_prompts, start)
            else:
                # Use propagated mask from previous chunk as prompt
                chunk_keyframe = 0  # Use first frame of chunk
                chunk_prompts = None  # Will be set from previous chunk's last mask
            
            chunks.append((start, end, chunk_keyframe, chunk_prompts))
            
            if end >= frame_count - 1:
                break
        
        return chunks

    def _adjust_prompts_for_chunk(self, prompts, chunk_start):
        """Adjust prompt frame indices relative to chunk start"""
        adjusted = []
        for prompt in prompts:
            new_prompt = prompt.copy()
            new_prompt["frame_idx"] = prompt["frame_idx"] - chunk_start
            if 0 <= new_prompt["frame_idx"]:  # Only include prompts within chunk
                adjusted.append(new_prompt)
        return adjusted

    def _process_chunk(self, seq_dir, start, end, chunk_keyframe, chunk_prompts, verbose):
        """Process a single chunk and return masks"""
        from pathlib import Path
        import tempfile
        import shutil
        
        # Create temporary directory for this chunk
        chunk_dir = Path(tempfile.mkdtemp(prefix=f"chunk_{start}_{end}_"))
        
        try:
            # Copy chunk frames to temporary directory
            self._create_chunk_sequence(seq_dir, chunk_dir, start, end)
            
            # Clear memory before processing
            clear_gpu_memory()
            
            # Initialize SAM2 state for this chunk
            inference_state = self.predictor.init_state(video_path=str(chunk_dir))
            self.predictor.reset_state(inference_state)
            
            # Add prompts or propagate from previous chunk
            if chunk_prompts:
                # Use original prompts (first chunk or chunk containing keyframe)
                self._add_prompts_to_state(inference_state, chunk_prompts, verbose)
            else:
                # TODO: Use mask from previous chunk as prompt
                # For now, skip chunks without prompts (will be handled by overlap merging)
                return [None] * (end - start + 1)
            
            # Track within chunk
            chunk_frame_count = end - start + 1
            all_masks_by_object = self._propagate_bidirectional(inference_state, chunk_frame_count, chunk_keyframe)
            
            # Extract masks
            chunk_masks = [None] * chunk_frame_count
            if 0 in all_masks_by_object:
                for f_idx, mask in all_masks_by_object[0].items():
                    if 0 <= f_idx < chunk_frame_count:
                        chunk_masks[f_idx] = mask
            
            return chunk_masks
            
        finally:
            # Clean up temporary chunk directory
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir)

    def _create_chunk_sequence(self, source_dir, chunk_dir, start, end):
        """Create a temporary sequence directory for the chunk"""
        import shutil
        
        source_path = Path(source_dir)
        jpgs = sorted([p for p in source_path.iterdir() if p.suffix.lower() == ".jpg"])
        
        # Copy frames for this chunk, renaming to 0-based sequence
        for i, frame_idx in enumerate(range(start, end + 1)):
            if frame_idx < len(jpgs):
                src_file = jpgs[frame_idx]
                dst_file = chunk_dir / f"{i:05d}.jpg"
                shutil.copy2(src_file, dst_file)

    def _merge_chunk_masks(self, final_masks, chunk_masks, start, end, overlap, verbose):
        """Merge chunk masks into final result with overlap handling"""
        for i, mask in enumerate(chunk_masks):
            global_idx = start + i
            if global_idx < len(final_masks):
                if final_masks[global_idx] is None:
                    # No existing mask, use chunk mask
                    final_masks[global_idx] = mask
                elif mask is not None:
                    # Both masks exist in overlap region - blend them
                    if overlap > 0 and i < overlap:
                        # In overlap region, blend masks
                        weight = i / overlap  # Fade in new chunk
                        final_masks[global_idx] = self._blend_masks(final_masks[global_idx], mask, weight)
                    else:
                        # Outside overlap, prefer new chunk
                        final_masks[global_idx] = mask

    def _blend_masks(self, mask1, mask2, weight):
        """Simple mask blending for overlap regions"""
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1
        
        # Convert to float for blending
        m1 = mask1.astype(np.float32)
        m2 = mask2.astype(np.float32)
        
        # Weighted blend
        blended = (1 - weight) * m1 + weight * m2
        
        # Convert back to boolean
        return (blended > 0.5).astype(bool)

    def _fill_occlusion_gaps(self, masks, inference_state, frame_count, threshold, verbose):
        """Detect and fill gaps caused by occlusions"""
        # Find gaps in tracking
        gaps = self._find_tracking_gaps(masks, threshold)

        if not gaps and verbose:
            print("🔍 No significant gaps found")
            return masks

        if verbose:
            print(f"🔍 Found {len(gaps)} gap(s) to fill:")
            for gap in gaps[:3]:  # Show first 3 gaps
                print(f"   Gap: frames {gap['start']}-{gap['end']} (reappears at {gap['reappear']})")

        # Fill each gap
        for gap in gaps:
            if verbose:
                print(f"🔧 Filling gap {gap['start']}-{gap['end']} from reappearance at {gap['reappear']}")

            # Reset state for this gap
            self.predictor.reset_state(inference_state)

            # Add the reappearance frame as a new prompt
            reappear_mask = masks[gap['reappear']]
            if reappear_mask is not None:
                # Convert mask to points (sample from mask boundary)
                points = self._mask_to_points(reappear_mask, n_points=10)
                if points:
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=gap['reappear'],
                        obj_id=0,
                        points=points,
                        labels=[1] * len(points)
                    )

                    # Propagate backwards to fill the gap
                    gap_masks = {}
                    for frame_idx, obj_ids, mask_tensor in self.predictor.propagate_in_video(
                            inference_state,
                            reverse=True,
                            start_frame_idx=gap['reappear'],
                            max_frame_num_to_track=gap['reappear'] - gap['start'] + 1
                    ):
                        if frame_idx < gap['start']:
                            break
                        if frame_idx <= gap['end']:
                            for i, obj_id in enumerate(obj_ids):
                                if obj_id == 0:
                                    mask_bool = self._to_prob_and_binarize(mask_tensor[i])
                                    gap_masks[frame_idx] = mask_bool

                    # Update masks with filled gaps
                    for f_idx, mask in gap_masks.items():
                        if masks[f_idx] is None:
                            masks[f_idx] = mask
                            if verbose:
                                print(f"   ✓ Filled frame {f_idx}")

        return masks

    def _find_tracking_gaps(self, masks, threshold):
        """Find gaps where object disappears and reappears"""
        gaps = []
        in_gap = False
        gap_start = None
        last_valid = None

        # Calculate average mask area for reference
        mask_areas = []
        for mask in masks:
            if mask is not None:
                area = np.sum(mask)
                if area > 0:
                    mask_areas.append(area)

        if not mask_areas:
            return gaps

        avg_area = np.mean(mask_areas)
        min_area = avg_area * threshold

        for i, mask in enumerate(masks):
            if mask is None or np.sum(mask) < min_area:
                # Object lost
                if not in_gap and last_valid is not None:
                    in_gap = True
                    gap_start = i
            else:
                # Object present
                if in_gap:
                    # Gap ended - object reappeared
                    gaps.append({
                        'start': gap_start,
                        'end': i - 1,
                        'reappear': i,
                        'last_valid_before': last_valid
                    })
                    in_gap = False
                last_valid = i

        return gaps

    def _mask_to_points(self, mask, n_points=10):
        """Convert mask to boundary points for re-initialization"""
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Sample points from contour
        if len(largest_contour) < n_points:
            points = [(int(p[0][0]), int(p[0][1])) for p in largest_contour]
        else:
            # Sample evenly
            indices = np.linspace(0, len(largest_contour) - 1, n_points, dtype=int)
            points = [(int(largest_contour[i][0][0]), int(largest_contour[i][0][1])) for i in indices]

        return points

    def _validate_image_sequence(self, path_like):
        """Ensure directory exists and contains at least one .jpg, all same size."""
        seq_dir = Path(path_like)
        if not seq_dir.is_dir():
            raise ValueError(f"Expected image-sequence directory, got: {seq_dir}")

        jpgs = sorted([p for p in seq_dir.iterdir() if p.suffix.lower() == ".jpg"])

        # If no jpgs found, also check for specific patterns like frame_*.jpg
        if len(jpgs) == 0:
            jpgs = sorted([p for p in seq_dir.iterdir()
                           if p.suffix.lower() == ".jpg" and p.name.startswith("frame_")])
        if len(jpgs) == 0:
            raise ValueError(f"No .jpg images found in {seq_dir}")

        # optional sanity checks
        first = cv2.imread(str(jpgs[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Failed to read first image: {jpgs[0]}")
        h0, w0 = first.shape[:2]

        # quick dimension consistency check on a handful of frames
        for p in [jpgs[0], jpgs[len(jpgs) // 2], jpgs[-1]]:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {p}")
            h, w = img.shape[:2]
            if (h, w) != (h0, w0):
                raise ValueError(f"Mixed resolutions in sequence. {jpgs[0].name}: {w0}x{h0}, {p.name}: {w}x{h}")

        # optional naming check for gaps
        try:
            nums = []
            for p in jpgs:
                stem = p.stem
                if stem.isdigit():
                    nums.append(int(stem))
            if len(nums) == len(jpgs):
                nums_sorted = sorted(nums)
                gaps = [i for i in range(nums_sorted[0], nums_sorted[-1] + 1) if i not in set(nums_sorted)]
                if gaps:
                    print(f"⚠️ Warning: non-contiguous frame numbers in {seq_dir}, gaps like {gaps[:5]} ...")
        except Exception:
            pass

        return seq_dir, len(jpgs)

    def _normalize_prompts(self, prompts, default_frame_idx, verbose):
        """Convert various prompt formats to normalized multi-frame format"""
        # Handle multi-frame format (already normalized)
        if isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], dict):
            if verbose:
                print(f"📋 Using multi-frame prompts: {len(prompts)} prompts")
            return prompts

        # Auto-detect single prompt format
        if isinstance(prompts, tuple) and len(prompts) == 2:
            # Single point: (x, y)
            if verbose:
                print(f"📍 Single point prompt at frame {default_frame_idx}")
            return [{"frame_idx": default_frame_idx, "points": [prompts], "labels": [1]}]

        elif isinstance(prompts, list) and len(prompts) == 4 and all(isinstance(x, (int, float)) for x in prompts):
            # Box: [x1, y1, x2, y2]
            if verbose:
                print(f"📦 Box prompt at frame {default_frame_idx}")
            return [{"frame_idx": default_frame_idx, "box": prompts}]

        elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], tuple):
            # Multiple points: [(x1, y1), (x2, y2), ...]
            if verbose:
                print(f"📍 Multiple points ({len(prompts)}) at frame {default_frame_idx}")
            return [{"frame_idx": default_frame_idx, "points": prompts, "labels": [1] * len(prompts)}]

        else:
            raise ValueError(f"Unsupported prompt format: {prompts}")

    def _add_prompts_to_state(self, state, prompts, verbose):
        """Add all prompts to the SAM2 state"""
        for prompt in prompts:
            frame_idx = prompt["frame_idx"]

            if "points" in prompt:
                points = prompt["points"]
                labels = prompt.get("labels", [1] * len(points))

                # Auto-expand labels if needed
                if len(labels) < len(points):
                    last_label = labels[-1] if labels else 1
                    labels = labels + [last_label] * (len(points) - len(labels))
                    if verbose:
                        print(f"🔄 Expanded labels to {len(labels)} for {len(points)} points")

                _, _, masks = self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=0,  # Always single object
                    points=points,
                    labels=labels
                )
                if verbose:
                    print(f"✅ Added {len(points)} point(s) at frame {frame_idx}, got {len(masks)} masks")

            elif "box" in prompt:
                box = prompt["box"]
                _, _, masks = self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=frame_idx,
                    obj_id=0,  # Always single object
                    box=box
                )
                if verbose:
                    print(f"✅ Added box at frame {frame_idx}, got {len(masks)} masks")

    def _propagate_bidirectional(self, state, total_frames, keyframe_idx):
        all_masks_by_object = {}
        print(f"🔄 Bidirectional propagation from keyframe {keyframe_idx} 0..{total_frames - 1}")

        # forward
        print("  → Forward propagation...")
        for frame_idx, obj_ids, mask_tensor in self.predictor.propagate_in_video(state, reverse=False):
            for i, obj_id in enumerate(obj_ids):
                mask_bool = self._to_prob_and_binarize(mask_tensor[i])
                bucket = all_masks_by_object.setdefault(obj_id, {})
                prev = bucket.get(frame_idx)
                bucket[frame_idx] = mask_bool if prev is None else np.logical_or(prev, mask_bool)

        # backward
        print("  ← Backward propagation...")
        for frame_idx, obj_ids, mask_tensor in self.predictor.propagate_in_video(state, reverse=True):
            for i, obj_id in enumerate(obj_ids):
                mask_bool = self._to_prob_and_binarize(mask_tensor[i])
                bucket = all_masks_by_object.setdefault(obj_id, {})
                prev = bucket.get(frame_idx)
                bucket[frame_idx] = mask_bool if prev is None else np.logical_or(prev, mask_bool)

        return all_masks_by_object

    def _to_prob_and_binarize(self, mask_tensor, threshold=0.0):
        """Convert SAM2 mask tensor to binary numpy array"""
        if hasattr(mask_tensor, 'cpu'):
            # PyTorch tensor
            mask_np = mask_tensor.cpu().numpy()
        else:
            # Already numpy
            mask_np = mask_tensor

        # Remove batch dimension if present
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]

        # Binarize
        return (mask_np > threshold).astype(bool)

    def calculate_union_mask(self, masks):
        """Calculate union of all masks"""
        union = None
        for mask in masks:
            if mask is not None:
                if union is None:
                    union = mask.copy()
                else:
                    union = np.logical_or(union, mask)
        return union

    def calculate_occlusions(self, masks, keyframe_idx):
        """Calculate occlusion masks"""
        if keyframe_idx >= len(masks) or masks[keyframe_idx] is None:
            return None

        keyframe_mask = masks[keyframe_idx]
        occlusions = []

        for i, mask in enumerate(masks):
            if mask is not None:
                # Occlusion = keyframe mask - current mask
                occlusion = np.logical_and(keyframe_mask, np.logical_not(mask))
                occlusions.append(occlusion)
            else:
                occlusions.append(None)

        return occlusions

def _extract_frames(video_path: str, frames_dir: Path,
                    frame_start: int = None, frame_end: int = None):
    """Extract frames from video - same logic as license plate pipeline"""
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set frame range
    start_frame = frame_start if frame_start is not None else 0
    end_frame = frame_end if frame_end is not None else total_frames - 1

    # Validate range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))


    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    extracted_count = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

            frame_path = frames_dir / f"{extracted_count:06d}.jpg"  # 0-based indexing
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1
        extracted_count += 1

    cap.release()
    return extracted_count, width, height


def main():
    """Demo function showing different tracking scenarios"""
    print("🚀 SAM2 ROI Tracker Demo - Enhanced Version")

    # Load SAM2 with preprocessing enabled
    samuel = load_sam2_local(predictor_type="video", model_size="base")
    roi_tracker = ROITracker(samuel)

    video_path = "Shot_4.mov"
    multi_frame_prompts = [
        # {"frame_idx": 10,
        #  "points": [(1209, 999), (1214, 993), (1224, 1020), (1213, 1029), (1225, 1037),  (1190, 1024), (1189, 974), (1224, 963), (1245, 981), (1245, 1019), (1216, 1065), (944, 935), (953, 989)],
        #  "labels": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
        # {"frame_idx": 148,
        #  "points": [(1452, 984), (1459, 975), (1478, 1002), (1462, 1017),
        #             (1423, 976), (1425, 951), (1447, 926), (1489, 926), (1507, 1005), (1477, 1050), (1161, 937),
        #             (1249, 781)],
        #  "labels": [1, 1, 1, 1,
        #             0, 0, 0, 0, 0, 0, 0, 0]}
        {"frame_idx": 80,
         "points": [(1618, 1010), (1657, 1012), (1677, 1004), (1734, 1007), (1720, 1011), (1661, 1010),
                    (1621, 1029), (1701, 1030), (1761, 1008), (1716, 985), (1610, 985), (1580, 999)],
         "labels": [1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0]}

    ]

    _extract_frames(video_path, Path("Shot_4/frames"))

    masks4, output4 = roi_tracker.track(
        video_path="Shot_4/frames",
        prompts=multi_frame_prompts,
        save_to="Shot_4/masks",
    )
    print(f"📊 Generated {len([m for m in masks4 if m is not None])} masks")


if __name__ == "__main__":
    main()