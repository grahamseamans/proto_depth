import math
import numpy as np
from tinygrad.nn import Module, Linear
from tinygrad import Tensor, nn

# Assume you have:
# - A resnet module: resnet(x) -> (B,2048,H',W')
# - chamfer_distance(pred, target) -> returns (loss, None)
# - rotation_6d_to_matrix(rot_6d) -> returns rotation matrices


class SceneEncoder(Module):
    def __init__(
        self,
        num_slots=32,
        num_prototypes=50,
        feature_dim=2048,
        num_points_per_proto=512,
        focal=847.630211643,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_prototypes = num_prototypes
        self.focal = focal
        self.feature_dim = feature_dim
        self.num_points_per_proto = num_points_per_proto

        # Assume resnet is already defined outside
        # self.resnet = resnet  # Provided externally

        # Camera head: 9 params (3 trans + 6D rotation)
        self.camera_head = Linear(feature_dim, 9)

        # Slot head: each slot => proto_selection (num_prototypes) + 3 trans + 6 rot + 1 scale
        out_dim = self.num_slots * (self.num_prototypes + 10)
        self.slot_head = Linear(feature_dim, out_dim)

        # Initialize prototypes as spheres
        # We'll use uniform random angles:
        # phi in [0,2pi), costheta in [-1,1]
        # x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = costheta
        # theta = arccos(costheta)
        proto_list = []
        for _ in range(self.num_prototypes):
            phi = Tensor(np.random.rand(self.num_points_per_proto)).mul(2 * math.pi)
            costheta = Tensor(np.random.rand(self.num_points_per_proto)) * 2 - 1
            sintheta = (1 - costheta**2).sqrt()
            x = sintheta * phi.cos()
            y = sintheta * phi.sin()
            z = costheta
            sphere_points = Tensor.stack([x, y, z], axis=-1)
            proto_list.append(sphere_points)
        prototypes = Tensor.stack(
            proto_list, axis=0
        )  # (num_prototypes, num_points_per_proto,3)
        prototypes.requires_grad = True
        self.prototypes = prototypes

    def forward(self, feat):
        # feat: (B, 2048)
        camera_params = self.camera_head(feat)  # (B,9)
        slot_out = self.slot_head(feat)  # (B, num_slots*(num_prototypes+10))

        B = feat.shape[0]
        slot_out = slot_out.reshape(B, self.num_slots, self.num_prototypes + 10)

        proto_logits = slot_out[:, :, : self.num_prototypes]  # (B,S,num_prototypes)
        trans = slot_out[:, :, self.num_prototypes : self.num_prototypes + 3]  # (B,S,3)
        rot_6d = slot_out[
            :, :, self.num_prototypes + 3 : self.num_prototypes + 9
        ]  # (B,S,6)
        scale = slot_out[
            :, :, self.num_prototypes + 9 : self.num_prototypes + 10
        ]  # (B,S,1)

        proto_probs = proto_logits.softmax(axis=-1)
        return camera_params, proto_probs, trans, rot_6d, scale


def select_and_transform_prototypes(model, proto_probs, trans, rot_6d, scale):
    B, S, _ = proto_probs.shape
    # Argmax to select prototype index per slot
    proto_idx = proto_probs.argmax(axis=-1)  # (B,S)

    # Gather prototypes
    # model.prototypes: (num_prototypes, num_points_per_proto, 3)
    chosen_protos_list = []
    for b in range(B):
        these_indices = (
            proto_idx[b].numpy().astype(int)
        )  # convert to numpy for indexing
        p = model.prototypes[these_indices]  # (S,num_points_per_proto,3)
        chosen_protos_list.append(p)
    chosen_protos = Tensor.stack(chosen_protos_list, axis=0)  # (B,S,P,3)

    # Convert rotations
    # rot_6d: (B,S,6)
    B_, S_ = rot_6d.shape[0], rot_6d.shape[1]
    rot_mats = rotation_6d_to_matrix(rot_6d.reshape(B_ * S_, 6)).reshape(B_, S_, 3, 3)

    scale_ = scale.reshape(B, S, 1, 1)
    trans_ = trans.reshape(B, S, 1, 3)

    # scale
    chosen_protos = chosen_protos * scale_

    # rotate (B,S,P,3) x (B,S,3,3) => (B,S,P,3)
    # tinygrad broadcasting works similarly to numpy
    chosen_protos = chosen_protos.dot(
        rot_mats
    )  # dot handles last dimension multiplication

    # translate
    chosen_protos = chosen_protos + trans_

    # reshape to (B, S*P, 3)
    B, S, P, _ = chosen_protos.shape
    final_pcl = chosen_protos.reshape(B, S * P, 3)
    return final_pcl


def train_one_epoch(model, dataloader, optimizer, device):
    model.training = True
    total_loss = 0.0
    for depth_img_3ch, target_points in dataloader:
        depth_img_3ch, target_points = (
            depth_img_3ch.to(device),
            target_points.to(device),
        )

        # forward resnet (assume resnet: depth_img_3ch->(B,2048,H',W'))
        features = resnet(depth_img_3ch)
        # AdaptiveAvgPool2d((1,1)) -> mean over H',W'
        features = features.mean(axis=(2, 3))  # (B,2048)
        camera_params, proto_probs, trans, rot_6d, scale = model(features)

        pred_points = select_and_transform_prototypes(
            model, proto_probs, trans, rot_6d, scale
        )

        # chamfer_distance returns (loss, _) => loss is a Tensor
        # ensure target_points shape is (B,M,3)
        if len(target_points.shape) == 2:
            target_points = target_points.unsqueeze(0)
        loss, _ = chamfer_distance(pred_points, target_points)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().numpy()

    return total_loss / len(dataloader)


def main():
    # Setup data loader, model, optimizer
    # Assume dataset, dataloader return Tinygrad Tensors
    dataset = SynthiaDepthDataset(...)  # You must define similarly as before
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    device = "cuda"  # or "cpu"
    model = SceneEncoder(
        num_slots=32, num_prototypes=50, feature_dim=2048, num_points_per_proto=512
    ).to(device)
    optimizer = nn.optim.Adam(nn.state.get_parameters(model))
    # optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Save model parameters if needed
    # Tinygrad doesn't have a built-in save. Just np.save them:
    params = [p.numpy() for p in model.parameters()]
    np.save("scene_encoder_params.npy", params)


if __name__ == "__main__":
    main()
