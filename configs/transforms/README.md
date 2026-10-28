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
| `digits.yml` | crop, rotation up to 15 degrees, brightness / contrast jitter (one channel) | grey handwritten digits (MNIST) |
| `digits_affine.yml` | `digits` without the jitter, plus slant (shear 15 degrees), shift (2 pixels) and zoom (0.9 to 1.1) | variant of `digits` for handwriting (docs 5.37) |
| `digits_affine_stroke.yml` | `digits_affine` + random stroke thickness | variant of `digits` for handwriting (docs 5.37) |
| `digits_zoomshift.yml` | no crop, no jitter: rotation and shear 8 degrees, zoom 0.9 to 1.1, shift 1.4 pixels, per step | variant 3 of `digits` for handwriting (docs 5.37) |
| `digits_zoomshift_stroke.yml` | `digits_zoomshift` + random stroke thickness | variant 4 of `digits`: isolates the stroke width on the zoomshift base (docs 5.37) |
| `digits_full.yml` | `digits_zoomshift_stroke` + elastic distortion, light blur, noise, 4 x 4 erasing | variant 5: every augmentation that keeps the digit, all at once (docs 5.39); worse than variant 4 on seed 0 |
| `digits_zoomshift_stroke_elastic.yml` | `digits_zoomshift_stroke` + elastic distortion alone | variant 6: **LOCKED reference bank for MNIST** (docs 5.57); 5 views per orbit, `d_max` 2 |
| `digits_zoomshift_stroke_elastic_erasing.yml` | `digits_zoomshift_stroke_elastic` + 4 x 4 erasing alone | variant 7a: is the erasing a culprit of `digits_full`? (docs 5.46) |
| `digits_zoomshift_stroke_elastic_noise.yml` | `digits_zoomshift_stroke_elastic` + gaussian noise alone | variant 7b: is the noise a culprit of `digits_full`? (docs 5.46) |
| `digits_zoomshift_stroke_elastic_noise01.yml` | same as 7b with max std 0.1 instead of 0.04 | variant 7c: does a stronger noise help the invariance? launched only if 7b recovers (user, 2026-10-26; docs 5.50) |
| `digits_zoomshift_stroke_elastic_blur.yml` | `digits_zoomshift_stroke_elastic` + light gaussian blur alone | variant 7d: is the blur a culprit of `digits_full`? run only if `noise01` collapses clearly (user, 2026-10-27; docs 5.50) |
| `digits_strong.yml` | `digits` with a stronger crop (area >= 0.5), rotation up to 30 degrees, jitter 0.3 | sensitivity step of `digits`, not a replacement |

## Locked bank for MNIST (docs 5.57, 2026-10-28)

The bank of every later MNIST experiment (baselines, ablations, Colored MNIST) is `digits_zoomshift_stroke_elastic.yml`
with `views_per_orbit: 5` and `d_max: 2`. It is not tuned any further. `digits_full`, the erasing, the noise (0.04 and 0.1)
and the blur were tried on top of it and are not part of the bank: the files stay in the folder to document what was tested.

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
