# Project Plan: Depthmap to 3d scene encoder.

<img width="322" alt="GneOUObbQAA3f2S" src="https://github.com/user-attachments/assets/294bf7d2-93d8-4ee4-a16a-4a8338abef53" />

## Phase One: Encoder from Visual to 3D

### Dataset
- Use the SYNTHIA dataset or similar.
- Preprocess the depth maps into point clouds for alignment during training.

### Goals
- Develop a system to encode a depth map into 3D scene representations:
  - **Prototypes:** Differentiable 3D meshes representing objects.
  - **Object Parameters:** Each object is parameterized by scale, position, orientation, and prototype weights.

### Steps
1. **Input and Network Design**
   - Input: Depth map.
   - **Encoder Architecture:** 
     - Use a ResNet or similar to extract a latent vector for each object slot.
     - Decompose as follows:
       - Scale: `scale = vecta[:1]`
       - Transform (Position & Orientation): `transform = vecta[1:7]` (x, y, z, yaw, pitch, roll)
       - Prototype Weights: `logits = vecta[7:x]`
   - Apply softmax to logits to assign a weighted combination of prototypes, allowing smooth interpolation between prototypes.

2. **Prototype Handling**
   - Define a bank of differentiable 3D meshes (prototypes).
   - **Blurring Prototypes:** interpolate between prototypes based on softmax weights to enable gradient flow.

3. **Scene Reconstruction**
   - For each object:
     - Select and blend prototypes using the softmax-weighted combination of `logits`.
     - Transform prototypes using `scale` and `transform`.
   - Render for fun.
   - Really each object is a mesh transform - we actually just have one true object (the sphere) which doesn't get backpropped, but we have all of these mesh transforms.

   when we make linear combo we softmax the logits, then use that to scale how much of each transform is used on the sphere

4. **Loss Function**
   - Use one directional point cloud loss (e.g., Chamfer Distance):
     - Compare the transformed meshes with the point cloud derived from the input depth map.
     - For each point in the depth map's point cloud, find the closest point on any mesh and bring it closer.

5. **Output**
   - Train the network to minimize loss, learning prototypes and how to use them to reconstruct 3D scenes from depth maps.

---

## Phase Two: Autoregressive Model for Video Sequences

### Dataset
- Use SYNTHIA's video sequences for temporal data with camera motion and object dynamics.

### Goals
- Extend the system to process video input, learning object trajectories and enforcing temporal consistency.

### Steps
1. **Slot-Based Representation**
   - Represent each object slot with a vector for each frame.
   - Penalize changes in prototype weights to maintain object consistency across frames.

2. **Temporal Regularization**
   - **Motion Loss:** Apply regularization for natural motion patterns, such as:
     - Parabolic trajectories for object movement.
     - Smooth transitions in object transformations (scale, position, orientation).
   - **Loss Application:** Apply motion loss to all object transformations

3. **Camera Integration** (optional as motion is relative, but might help)
   - Add a **camera vector** (position and rotation) to model relative camera motion.
   - Use the camera vector to align object transformations with the global frame.

4. **Loss Function**
   - **Temporal Consistency Loss:** Penalize abrupt changes in object parameters across frames.
   - **Depth Alignment:** Continue using depth map and point cloud losses for each frame.

5. **Output**
   - Train the network to reconstruct consistent 3D scenes across video frames.

---

## Stretch Goals

### 1. Movement Cycles for Prototypes
- Add specific motion cycles (e.g., wheel turning, walking) for manipulating prototypes.
- Incorporate motion cycle parameters into the object vectors.

### 2. Associating Prototypes with Words
- Map prototypes to nouns and motions to verbs for semantic understanding.
- Use language models to integrate text annotations with visual data.

### 3. Coupling Objects
- Learn relationships between objects, such as:
  - A person entering a car and moving with it.
- Model coupled dynamics in the loss function.

### 4. Prototypical Textures
- Extend the system to learn textures (e.g., colors, fonts) for prototypes.
- Render scenes with both 3D structure and textural realism.

### 4. Hierarcical protoypes
- Something more like this: (https://arxiv.org/pdf/1905.05622)
- Wheels are on bikes, cars... find a way to have these objects share prototypes.

if the autoregressive thing works, i can copy deepseek grpo, make time seperate from steps, and then have it rl try to get to later frames.

so like give it 10 frames, then ask it what it thinks is going to happen at t + 10, maybe with a few guesses and then reward it if any of them are right

it can take as many steps as it wants, so it would need to like, train in a weird way where it wasn't time locked.

(to clarify here, the idea is that with a sequence of x inputs, they're not linearly spaced over time with 1/x being the time difference between steps)

but if it can learn that each of it's inputs aren't time locked, and that time itself is some, like, seperate quality, that can move forwards and backwards, 

then we're doing great

we could train it will all sort of sequences, with increasing and decreasing rate of time

fast rate of time, slow rate of time.

time would need to be an input, like the depth map, idk what time would be, a string? utc? relative (sine waves again?) idk, it's different from positional embedding though.
