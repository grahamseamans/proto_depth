import os
import random
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

# If needed, adjust these constants or put them as arguments.
FOCAL = 847.630211643
MAX_DEPTH = 1000.0
CUTOFF_DISTANCE = 300.0


def load_depth_image(depth_path):
    im = Image.open(depth_path)
    im = np.array(im, dtype=np.int32)
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]
    depth_int24 = R + G * 256 + B * (256**2)
    depth = depth_int24.astype(np.float64) / ((256**3) - 1) * 1000.0  # meters
    return depth


def normalize_depth(depth):
    # sqrt scaling:
    ratio = depth / CUTOFF_DISTANCE
    normalized = np.sqrt(np.clip(ratio, 0, 1))
    return normalized


def depth_to_pointcloud(depth, focal=FOCAL, max_depth=MAX_DEPTH):
    H, W = depth.shape
    cx = W / 2.0
    cy = H / 2.0
    fx = focal
    fy = focal

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    valid_mask = points[:, 2] < max_depth
    points = points[valid_mask]
    return points


def transform_fn(data_item):
    # data_item is a tuple (depth_path,)
    depth_path = data_item[0]

    depth = load_depth_image(depth_path)
    depth_normalized = normalize_depth(depth)

    # Convert to Tensors
    depth_img = torch.from_numpy(depth_normalized[None].astype(np.float32))  # (1,H,W)
    depth_img_3ch = depth_img.expand(3, -1, -1)  # (3,H,W)

    points = depth_to_pointcloud(depth)
    points_t = torch.from_numpy(points.astype(np.float32))  # (N,3)

    # return (input, target) format
    return (depth_img_3ch, points_t, None)


###################################
# Integration with DataHandler
###################################
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class DataHandler:
    def __init__(
        self,
        data,
        test_ratio=0.2,
        seed=42,
        shuffle=True,
        transform_fn=None,
        num_workers=None,
        pool_type="thread",
    ):
        if transform_fn is None:
            raise ValueError("You must provide a transform_fn.")

        self.transform_fn = transform_fn
        self.shuffle = shuffle

        self.executor = None
        if num_workers is not None and num_workers > 0:
            if pool_type == "thread":
                self.executor = ThreadPoolExecutor(max_workers=num_workers)
            else:
                self.executor = ProcessPoolExecutor(max_workers=num_workers)

        self.train_data, self.test_data = self.train_test_split(data, test_ratio, seed)

        self.train_samples = self.Samples(len(self.train_data), shuffle=shuffle)
        self.test_samples = self.Samples(len(self.test_data), shuffle=False)

    @staticmethod
    def train_test_split(data, test_ratio=0.2, seed=42):
        random.seed(seed)
        data_shuffled = data[:]
        random.shuffle(data_shuffled)
        test_size = int(len(data_shuffled) * test_ratio)
        test_data = data_shuffled[:test_size]
        train_data = data_shuffled[test_size:]
        return train_data, test_data

    class Samples:
        def __init__(self, length, shuffle=True):
            self.length = length
            self.shuffle = shuffle
            self.sample_idxs = list(range(length))
            if self.shuffle:
                random.shuffle(self.sample_idxs)

        def idxs(self, batch_size):
            assert batch_size <= self.length
            if len(self.sample_idxs) < batch_size:
                self.sample_idxs = list(range(self.length))
                if self.shuffle:
                    random.shuffle(self.sample_idxs)
            ret = self.sample_idxs[:batch_size]
            self.sample_idxs = self.sample_idxs[batch_size:]
            return ret

        def reset(self):
            self.sample_idxs = list(range(self.length))
            if self.shuffle:
                random.shuffle(self.sample_idxs)

    def make_batch(self, data, idxs):
        if self.executor:
            processed = list(
                self.executor.map(self.transform_fn, [data[i] for i in idxs])
            )
        else:
            processed = [self.transform_fn(data[i]) for i in idxs]

        # processed is a list of tuples like (depth_img_3ch, points_t, ann_index)
        # transpose them
        transposed = list(zip(*processed))

        out = []
        for elements in transposed:
            # elements is a tuple of length batch_size
            # if elements are all Tensor, stack them
            if isinstance(elements[0], torch.Tensor):
                out.append(torch.stack(elements))
            else:
                # just return as a list (e.g. ann indexes)
                out.append(elements)

        return tuple(out)

    def get_train_batch(self, batch_size):
        idxs = self.train_samples.idxs(batch_size)
        return self.make_batch(self.train_data, idxs)

    def get_test_batch(self, batch_size):
        idxs = self.test_samples.idxs(batch_size)
        return self.make_batch(self.test_data, idxs)

    def reset_train(self):
        self.train_samples.reset()

    def reset_test(self):
        self.test_samples.reset()

    def close(self):
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

    def __del__(self):
        self.close()


def stratified_depth_sampling(points, bins=5, samples_per_bin=2000):
    z_vals = points[:, 2]
    min_z, max_z = z_vals.min(), z_vals.max()
    bin_edges = np.linspace(min_z, max_z, bins + 1)
    sampled_points = []

    for i in range(bins):
        bin_mask = (z_vals >= bin_edges[i]) & (z_vals < bin_edges[i + 1])
        bin_points = points[bin_mask]
        if len(bin_points) == 0:
            continue

        if len(bin_points) > samples_per_bin:
            idxs = np.random.choice(len(bin_points), samples_per_bin, replace=False)
            bin_points = bin_points[idxs]
        sampled_points.append(bin_points)

    if len(sampled_points) > 0:
        sampled_points = np.concatenate(sampled_points, axis=0)
    else:
        sampled_points = points

    return sampled_points


def display_data(depth_img_3ch, points, depth_norm):
    pts = points[0].numpy()
    print(pts.shape)

    pts_sample = stratified_depth_sampling(pts, bins=5, samples_per_bin=2000)

    fig = plt.figure(figsize=(15, 7))

    # Left subplot: depth image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(depth_norm[0], cmap="gray")
    ax1.set_title("Depth Image (Sqrt-Scaled)")
    ax1.set_xlabel("Image X")
    ax1.set_ylabel("Image Y")
    cbar1 = fig.colorbar(
        ax1.imshow(depth_norm[0], cmap="gray"), ax=ax1, fraction=0.046, pad=0.04
    )
    cbar1.set_label("Normalized Depth (sqrt scaling)")

    # Right subplot: 3D point cloud
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    scatter_plot = ax2.scatter(
        pts_sample[:, 0],
        pts_sample[:, 1],
        pts_sample[:, 2],
        s=1,
        c=pts_sample[:, 2],
        cmap="viridis",
    )
    fig.colorbar(scatter_plot, ax=ax2, label="Z depth (m)")
    ax2.set_title("Stratified Sampled Point Cloud")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.text2D(
        0.05,
        0.95,
        "Camera at Origin\nLooking along +Z",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", fc="w"),
    )

    plt.tight_layout()
    plt.savefig("debug_figure_sqrt_scale.png", dpi=150)
    print("Figure saved to debug_figure_sqrt_scale.png")
    plt.show()


#########################################
# Example usage
#########################################
if __name__ == "__main__":
    # Suppose you have a list of depth image paths:
    data_dir = "/Users/cibo/code/proto_depth/data/SYNTHIA-SF"
    # You can glob them:
    from glob import glob

    sequences = [1, 2, 3, 4, 5, 6]
    depth_files = []
    for seq in sequences:
        seq_dir = os.path.join(data_dir, f"SEQ{seq}", "DepthLeft")
        depth_files += [(path,) for path in glob(os.path.join(seq_dir, "*.png"))]

    # Now depth_files is a list of tuples, each with one element (the path)
    handler = DataHandler(data=depth_files, transform_fn=transform_fn, num_workers=4)

    batch = handler.get_train_batch(2)  # get a batch of size 2
    # batch should be (depth_imgs, points, ann_indexes)
    depth_imgs, points, ann_indexes = batch

    display_data(depth_imgs, points, depth_imgs[0].numpy())

    print(depth_imgs.shape)  # Should be (2,3,H,W)
    print(
        points.shape
    )  # Should be (2,N,3) where N can vary if not equal length, might need padding
    print(len(ann_indexes))  # 2 annoy indexes or None placeholders
