# flowers — a neural network from scratch

A two-input, one-output neuron written with nothing but NumPy, trained to tell
red flowers from blue ones using only their length and width.

Worked through for **Foundation of Vision & Language (FoVL)**, IIT Jodhpur —
the "Intro to NN" session. Based on the notebook shared in class
([JonComo/flowers](https://github.com/JonComo/flowers)), with the class fixes:
`train()` returns its weights instead of leaking globals, and each plot gets an
explicit `plt.show()`.

Anurag Mor · M26AI1023

## What's in the notebook

[`flowers.ipynb`](flowers.ipynb) — all cells executed, outputs and plots included.

| Step | What happens |
|---|---|
| Data | 8 labelled flowers, `[length, width, type]`, plus one mystery flower at `[4.5, 1]` |
| Model | `z = w1·length + w2·width + b`, squashed through a sigmoid |
| Loss | Squared error, `(pred - target)²` |
| Training | 10,000 steps of stochastic gradient descent, learning rate 0.1, gradients derived by hand via the chain rule |
| Result | The mystery flower scores **0.99** → red |

The last cell sweeps the whole x/y plane through the trained network, which
draws the decision boundary as a clean diagonal split with every training point
on the correct side.

## The gradient, by hand

No autograd here — the chain rule is written out term by term, which is the
whole point of the exercise:

```
dcost/dpred = 2 * (pred - target)
dpred/dz    = sigmoid'(z)
dz/dw1      = length,   dz/dw2 = width,   dz/db = 1

dcost/dw1   = dcost/dpred * dpred/dz * dz/dw1
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook flowers.ipynb
```

Training is randomly initialised, so the exact prediction shifts a little run to
run — it lands around 0.99 either way.
