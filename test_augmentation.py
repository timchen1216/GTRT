import numpy as np
import matplotlib.pyplot as plt
from mot_data_loader import (
    TrackletSplit,
    AddFP,
    BoxJitter,
    BoxShift,
    BoxClip,
    HFlip,
    RandomDelete,
)


def visualize_boxes(orig_boxes, aug_boxes, title, img_size=(1920, 1080)):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.ones(img_size))

    # Draw original boxes in red
    for box in orig_boxes:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
                label=(
                    "Original"
                    if "Original" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
        )
    # Draw augmented boxes in blue
    for box in aug_boxes:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="blue",
                linewidth=2,
                linestyle="--",
                label=(
                    "Augmented"
                    if "Augmented" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.axis("off")
    plt.show()


def create_sample_data():
    # Create a sample with 3 objects tracked over 5 frames
    sample = {
        "boxes": np.array(
            [
                [100, 100, 200, 200],  # frame 0, obj 1
                [150, 150, 250, 250],  # frame 1, obj 1
                [200, 200, 300, 300],  # frame 2, obj 1
                [300, 300, 400, 400],  # frame 0, obj 2
                [350, 350, 450, 450],  # frame 1, obj 2
                [400, 400, 500, 500],  # frame 2, obj 2
                [500, 500, 600, 600],  # frame 0, obj 3
                [550, 550, 650, 650],  # frame 1, obj 3
                [600, 600, 700, 700],  # frame 2, obj 3
            ]
        ),
        "gt_boxes": np.array(
            [
                [100, 100, 200, 200],  # frame 0, obj 1
                [150, 150, 250, 250],  # frame 1, obj 1
                [200, 200, 300, 300],  # frame 2, obj 1
                [300, 300, 400, 400],  # frame 0, obj 2
                [350, 350, 450, 450],  # frame 1, obj 2
                [400, 400, 500, 500],  # frame 2, obj 2
                [500, 500, 600, 600],  # frame 0, obj 3
                [550, 550, 650, 650],  # frame 1, obj 3
                [600, 600, 700, 700],  # frame 2, obj 3
            ]
        ),
        "obj_ids": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        "fr_ids": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        "classes": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "height": 1080,
        "width": 1920,
    }
    return sample


def test_augmentations():
    # Create sample data
    sample = create_sample_data()
    orig_boxes = sample["boxes"].copy()

    # Test TrackletSplit
    print("\nTesting TrackletSplit...")
    tracklet_split = TrackletSplit(p0=0.9, p1=0.95, p2=0.9)
    split_sample = tracklet_split(sample.copy())
    print(f"Original boxes count: {len(orig_boxes)}")
    print(f"After TrackletSplit boxes count: {len(split_sample['boxes'])}")
    visualize_boxes(orig_boxes, split_sample["boxes"], "TrackletSplit")

    # Test AddFP
    print("\nTesting AddFP...")
    add_fp = AddFP(temporal_len=3, fpr=0.2)
    fp_sample = add_fp(sample.copy())
    print(f"Original boxes count: {len(orig_boxes)}")
    print(f"After AddFP boxes count: {len(fp_sample['boxes'])}")
    visualize_boxes(orig_boxes, fp_sample["boxes"], "AddFP")

    # Test BoxJitter
    print("\nTesting BoxJitter...")
    box_jitter = BoxJitter(jitter_ratio=0.2)
    jitter_sample = box_jitter(sample.copy())
    print("Original first box:", orig_boxes[0])
    print("After BoxJitter first box:", jitter_sample["boxes"][0])
    visualize_boxes(orig_boxes, jitter_sample["boxes"], "BoxJitter")

    # Test BoxShift
    print("\nTesting BoxShift...")
    box_shift = BoxShift(shift_ratio=0.2)
    shift_sample = box_shift(sample.copy())
    print("Original first box:", orig_boxes[0])
    print("After BoxShift first box:", shift_sample["boxes"][0])
    visualize_boxes(orig_boxes, shift_sample["boxes"], "BoxShift")

    # Test BoxClip
    print("\nTesting BoxClip...")
    # Create a box that will be clipped
    clip_sample = sample.copy()
    clip_sample["boxes"][0] = [-100, -100, 2000, 2000]
    box_clip = BoxClip()
    clipped = box_clip(clip_sample)
    print("Original first box:", orig_boxes[0])
    visualize_boxes(orig_boxes, clipped["boxes"], "BoxClip")

    # Test HFlip
    print("\nTesting HFlip...")
    h_flip = HFlip(flip_ratio=1.0)  # Force flip for demonstration
    flip_sample = h_flip(sample.copy())
    print("Original first box:", orig_boxes[0])
    print("After HFlip first box:", flip_sample["boxes"][0])
    visualize_boxes(orig_boxes, flip_sample["boxes"], "HFlip")

    # Test RandomDelete
    print("\nTesting RandomDelete...")
    random_delete = RandomDelete(delete_ratio=0.3)
    delete_sample = random_delete(sample.copy())
    print(f"Original boxes count: {len(orig_boxes)}")
    print(f"After RandomDelete boxes count: {len(delete_sample['boxes'])}")
    visualize_boxes(orig_boxes, delete_sample["boxes"], "RandomDelete")


if __name__ == "__main__":
    test_augmentations()
