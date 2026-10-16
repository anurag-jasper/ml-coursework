# ML coursework

Notebooks from my M.Tech (AI & ML) coursework at IIT Jodhpur.

**Anurag Mor · M26AI1023**

| Notebook | Course | What it does |
|---|---|---|
| [`M26AI1023_FoVL_Individual_Project_CV.ipynb`](M26AI1023_FoVL_Individual_Project_CV.ipynb) | Foundations of Vision & Language | Measures how Gaussian blur and unsharp-mask sharpening affect YOLO11n detection accuracy on COCO128 |
| [`flowers.ipynb`](flowers.ipynb) | Foundations of Vision & Language | A two-input neuron built from scratch in NumPy, trained with hand-derived gradients |

Both notebooks are committed with all cells executed and outputs intact.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

---

## Effect of Blur and Sharpening on Object Detection

How much detection accuracy does a pretrained detector lose when its input is
blurred — and can sharpening win any of it back?

| | |
|---|---|
| Dataset | COCO128 (128 images, 929 labelled instances) |
| Model | Pretrained YOLO11n — **never retrained**, the same model object across all stages |
| Platform | Google Colab, CPU |

Only the *input images* change between stages. Labels and evaluation settings
are held fixed, so every difference in the table below is attributable to the
image transform alone.

| Stage | Transform |
|---|---|
| 1 — Baseline | Original COCO128 images, untouched |
| 2 — Blur | 9×9 Gaussian kernel (OpenCV `GaussianBlur`) |
| 3 — Sharpen | Unsharp mask, amount = 1.0, applied to the blurred images |

### Results

| Stage | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| 1 — Original / Baseline | 0.6633 | 0.5894 | 0.6696 | 0.5024 |
| 2 — Gaussian Blur (9×9) | 0.6784 | 0.5351 | 0.6168 | 0.4434 |
| 3 — Sharpened (unsharp mask) | 0.6192 | 0.5652 | 0.6296 | 0.4561 |

| Change | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Blur vs Baseline | +0.0151 | −0.0543 | −0.0528 | −0.0590 |
| Sharpened vs Blur | −0.0592 | +0.0301 | +0.0128 | +0.0127 |
| Sharpened vs Baseline | −0.0441 | −0.0242 | −0.0400 | −0.0463 |

### What the numbers say

**Blur costs recall, not precision.** mAP@0.5 drops 0.053 and recall drops
0.054, while precision actually ticks *up* slightly. Blurring erases the
high-frequency edges that small and low-contrast objects depend on, so the
detector simply stops proposing them — the boxes it does still emit are the
easy, high-confidence ones, which is why precision holds.

**Sharpening recovers a little, and not for free.** Running an unsharp mask over
the blurred images buys back +0.013 mAP@0.5 and +0.030 recall: some edge
structure is genuinely restored. But precision falls 0.059, because the same
operation amplifies noise into edge-like artefacts that trigger false positives.

**Sharpening does not undo blur.** Stage 3 still sits 0.040 mAP@0.5 below the
baseline. An unsharp mask re-weights the frequencies that survived the Gaussian;
it cannot restore detail the blur destroyed. The information is gone.

The notebook carries three validation checks beyond the brief: variance-of-
Laplacian sharpness measurements per stage, IoU-matched detection accounting at
a fixed confidence threshold, and a control run (original re-save, to rule out
JPEG re-encoding) plus a sharpening-strength sweep including a 3×3 Laplacian
kernel.

---

## Flowers — a neural network from scratch

A two-input, one-output neuron written with nothing but NumPy, trained to tell
red flowers from blue ones using only their length and width. From the "Intro to
NN" session, based on the notebook shared in class
([JonComo/flowers](https://github.com/JonComo/flowers)) with the class fixes:
`train()` returns its weights instead of leaking globals, and each plot gets an
explicit `plt.show()`.

| Step | What happens |
|---|---|
| Data | 8 labelled flowers, `[length, width, type]`, plus one mystery flower at `[4.5, 1]` |
| Model | `z = w1·length + w2·width + b`, squashed through a sigmoid |
| Loss | Squared error, `(pred - target)²` |
| Training | 10,000 steps of stochastic gradient descent, learning rate 0.1 |
| Result | The mystery flower scores **0.99** → red |

No autograd — the chain rule is written out term by term, which is the whole
point of the exercise:

```
dcost/dpred = 2 * (pred - target)
dpred/dz    = sigmoid'(z)
dz/dw1      = length,   dz/dw2 = width,   dz/db = 1

dcost/dw1   = dcost/dpred * dpred/dz * dz/dw1
```

The last cell sweeps the whole x/y plane through the trained network, drawing
the decision boundary as a clean diagonal split with every training point on the
correct side. Training is randomly initialised, so the exact prediction shifts a
little run to run — it lands around 0.99 either way.
