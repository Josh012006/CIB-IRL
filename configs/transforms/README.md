# Transformation banks (README Section 5.1)

A bank is the family T of image transformations that defines an ORBIT: the views of one image are declared to be
the same thing. T is not a robustness knob, it is the invariance we impose. Whatever T changes is a nuisance,
whatever it preserves is identity. A bank is therefore a written, versioned object, chosen BEFORE looking at results.

| Bank | Content | Use |
|---|---|---|
| `generic.yml` | crop, brightness / contrast / saturation jitter, rare grayscale; the hue is never touched | every dataset, unchanged: the main table |
| `natural.yml` | `generic` + horizontal flip | natural images only (a flipped digit or toy shape is another identity) |
| `informed_color.yml` | `generic` + full hue redraw | a DECLARED side information about the nuisance (colour): an experimental condition, not a tuning |
| `toy_oracle.yml` | exact redraw of the toy background | toy data only: an oracle upper bound on the prior |

Reading the two regimes on a dataset with a colour shortcut (Colored MNIST, toy with `spurious: 0.9`):
`generic` gives the model no prior about the nuisance, `informed_color` gives it one. The gap between the two
measures how much a result depends on that prior.

## Admission rule

An operation enters a bank only if domain knowledge, written down before any result is seen, says that it preserves
identity for the data the bank is used on. Never choose an operation, or its strength, by looking at the labels.
The strengths in the files are fixed in advance; their effect is studied by a sensitivity sweep (override files),
not by keeping the best value.

## File format

    name: generic                 # must match the file name
    description: ...
    ops:                          # applied in this order, to the output of the previous one
      - random_resized_crop: {scale: [0.7, 1.0], ratio: [0.8, 1.25]}
      - hue_rotate: {max_degrees: 180}
      - grayscale: {p: 0.1}

Operations are registered in `cibirl/transforms.py` (`available()` lists them); dataset-specific ones, such as
`toy_background_redraw`, are registered by `scripts/datasets.py`. The orbit sampler applies the whole bank up to
`d_max` times in a row, so strengths compound.
