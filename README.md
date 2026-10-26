# Video edit propagation with optical flow (RAFT)

Propagate an edit made on **one reference frame** to a whole video, automatically,
using dense **optical flow** estimated by a pretrained **RAFT** model (torchvision).
Evaluated on two sequences of the [DAVIS](https://davischallenge.org/) benchmark: *blackswan* and *camel*.

Course project, IMT Atlantique.

## Pipeline

1. **Load frames** from DAVIS (20 frames, 256×256), downloaded automatically.
2. **Pairwise optical flow** with RAFT (`raft_small`, pretrained), forward and backward,
   between consecutive frames.
3. **Long-term motion integration:** compose the pairwise flows to map every pixel of frame *t*
   back to its position in the reference frame.
4. **Edit on the reference frame:** the edit is stored as a residual
   $\Delta = I_0^{\text{edited}} - I_0$, restricted to the object mask (e.g. black swan → white swan).
5. **Propagation:** warp the residual to each frame with the composed flow,
   $I_t^{\text{edited}}(x) = I_t(x) + \Delta(\text{warp}_{t \to 0}(x))$.
6. **Occlusion handling:** forward–backward consistency check to avoid pasting the edit
   on pixels that are hidden.
7. **Outputs:** edited video, temporal-difference maps, failure cases.
8. **Evaluation** against the DAVIS ground-truth masks.

## Results

The propagated edit region is compared with the DAVIS ground-truth object mask in each frame
(mean / median / standard deviation over the 20 frames).

| Sequence | IoU | Dice / F1 | Boundary F1 |
|---|---|---|---|
| camel | **0.71** / 0.69 / 0.11 | **0.83** / 0.82 / 0.07 | 0.57 / 0.47 / 0.18 |
| blackswan | 0.61 / 0.54 / 0.17 | 0.74 / 0.70 / 0.12 | 0.49 / 0.38 / 0.22 |

**What the results show**
- The edit follows the object well overall (Dice 0.74–0.83), but **boundaries are much less
  accurate** (Boundary F1 ≈ 0.5): flow is blurred at motion boundaries.
- **Drift accumulates** along the sequence: each pairwise flow carries a small error, and
  chaining them makes the edit slide away from the object in later frames.
- The camel moves slowly and regularly, which likely explains its better scores; on blackswan, the
  water reflections and the thin neck plausibly make the flow less reliable.

## Notebooks

| Notebook | Sequence |
|---|---|
| `notebooks/raft_edit_propagation_blackswan.ipynb` | DAVIS *blackswan* |
| `notebooks/raft_edit_propagation_camel.ipynb` | DAVIS *camel* |

Designed for Google Colab with a GPU. The DAVIS frames and the pretrained RAFT weights are
downloaded automatically.

## Stack

Python · PyTorch · torchvision (RAFT) · OpenCV · NumPy · Matplotlib

## Limitations and next steps

- Only 20 frames per sequence; drift would grow on longer videos.
- The occlusion check is a coarse heuristic; re-anchoring on intermediate keyframes
  or using a segmentation model to correct the mask would reduce drift.
- `raft_large` was not compared with `raft_small`.
