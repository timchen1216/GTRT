import os
import numpy as np
import random


def process_mot_file(input_path, output_path):
    """Process MOT format file to remove duplicate IDs in same frame

    Args:
        input_path (str): Path to input MOT format file
        output_path (str): Path to save processed results
    """
    # Read all detections
    detections = []
    with open(input_path, "r") as f:
        for line in f:
            frame_id, track_id, x, y, w, h, conf, _, _, _ = map(
                float, line.strip().split(",")
            )
            detections.append([int(frame_id), int(track_id), x, y, w, h, conf])

    detections = np.array(detections)

    # Group by frame
    unique_frames = np.unique(detections[:, 0])
    filtered_detections = []

    # Track duplicate statistics
    total_duplicates = 0
    duplicate_frames = 0

    # Process each frame
    for frame in unique_frames:
        frame_mask = detections[:, 0] == frame
        frame_dets = detections[frame_mask]

        # Find duplicate IDs
        unique_ids, counts = np.unique(frame_dets[:, 1], return_counts=True)
        duplicate_ids = unique_ids[counts > 1]

        if len(duplicate_ids) > 0:
            duplicate_frames += 1
            total_duplicates += sum(counts[counts > 1] - 1)
            print(
                f"Frame {int(frame)}: Found {len(duplicate_ids)} duplicate IDs: {duplicate_ids}"
            )

        # Process duplicates
        keep_dets = []
        processed_ids = set()

        for det in frame_dets:
            track_id = det[1]
            if track_id in processed_ids:
                continue

            if track_id in duplicate_ids:
                # Get all detections with this ID in current frame
                id_mask = frame_dets[:, 1] == track_id
                same_id_dets = frame_dets[id_mask]
                # Calculate average position
                avg_det = same_id_dets.mean(axis=0)
                keep_dets.append(avg_det)
                processed_ids.add(track_id)
            else:
                keep_dets.append(det)
                processed_ids.add(track_id)

        filtered_detections.extend(keep_dets)

    print(f"\nSummary:")
    print(f"Total frames with duplicates: {duplicate_frames}")
    print(f"Total duplicate detections: {total_duplicates}")

    # Save results
    with open(output_path, "w") as f:
        for det in filtered_detections:
            f.write(
                f"{int(det[0])},{int(det[1])},{int(det[2])},{int(det[3])},{int(det[4])},{int(det[5])},1,-1,-1,-1\n"
            )

    return duplicate_frames, total_duplicates


def process_sequence(seq_name, input_dir, output_dir):
    """Process a single sequence

    Args:
        seq_name (str): Name of the sequence
        input_dir (str): Directory containing input files
        output_dir (str): Directory to save processed files
    """
    input_path = os.path.join(input_dir, f"{seq_name}.txt")
    output_path = os.path.join(output_dir, f"{seq_name}.txt")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing sequence: {seq_name}")
    process_mot_file(input_path, output_path)


def main():
    # You can modify these paths according to your needs
    input_dir = "save_txt/DanceTrack_test_T128_S5_DA"
    output_dir = "save_txt/DanceTrack_test_T128_S5_DA_rm"

    # Process all txt files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            seq_name = filename[:-4]
            process_sequence(seq_name, input_dir, output_dir)
            print(f"Processed {seq_name}")


if __name__ == "__main__":
    main()
