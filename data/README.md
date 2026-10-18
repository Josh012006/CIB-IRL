# Data

Only `data/toy/` and this file are tracked by git (see `.gitignore`). Section 13.2 of the main README gives the commands.

| Folder | Content | How to get it |
|---|---|---|
| `toy/` | The toy benchmark as files: seeds 0 to 9 by default, one `seedNNN.npz` per seed, and `manifest.json` (parameters, seeds, sha256 of the arrays of each file). | Tracked. More seeds: `python scripts/make_toy_dataset.py --seeds 0-199` (about 90 MB). Check: `python scripts/make_toy_dataset.py --verify`. |
| `mnist/` | MNIST idx files (train and t10k), used by the MNIST digits and by Colored MNIST. | Downloaded at the first use, md5-checked (network needed). |
| `waterbirds/` | Waterbirds: `metadata.csv` and the images. | By hand: the official release (WILDS `waterbirds_v1.0` or Group-DRO `waterbird_complete95_forest2water2`), about 1.8 GB. |

## Toy files

One file per seed, arrays of `float32` images in $[0,1]$ with shape `[N, 3, 16, 16]` (1500 images: 5 classes of 300), `int64` classes and `int64` coarse background-colour ids. The classes and ids are for evaluation only: a method never receives them.

| Arrays | Content | Generator |
|---|---|---|
| `train_{x,y,bg}` | the images a method clusters, background independent of the class (spurious 0) | `make_toy(seed=s, spurious=0)` |
| `train_s90_{x,y,bg}` | same classes, with the colour shortcut in training (spurious 0.9) | `make_toy(seed=s, spurious=0.9)` |
| `test_iid_{x,y,bg}` | new images, random backgrounds | `make_toy_test(s, s, "iid")` |
| `test_aligned_{x,y,bg}` | new images, background = palette colour of the class | `make_toy_test(s, s, "aligned")` |
| `test_inverted_{x,y,bg}` | new images, background = palette colour of another class | `make_toy_test(s, s, "inverted")` |

The generator (`scripts/toy_data.py`) stays the source of truth: the files hold exactly what it makes, and `tests/test_toy_files.py` and `--verify` check it.

```python
import sys; sys.path.insert(0, "scripts")
import datasets

train = datasets.build("toy", 0)                    # reads data/toy/seed000.npz; `train.source` is "file"
shortcut = datasets.build("toy", 0, spurious=0.9)   # the training split with the colour shortcut
test = datasets.build_test("toy", 0, "inverted")    # "iid", "aligned" or "inverted"
```

`datasets.build("toy", s, source="generate")` forces the generator; `source="file"` raises an error if the file or the split is missing. Any other configuration than 5 classes of 300 images at 16 x 16 is generated, never read from a file.
