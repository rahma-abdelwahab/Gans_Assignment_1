<<<<<<< HEAD
# DSAI 490 — Assignment 1: Representation Learning with Autoencoders

**Dataset:** Medical MNIST (`andrewmvd/medical-mnist` on Kaggle)  
**Regions:** AbdomenCT · BreastMRI · ChestCT · CXR · Hand · HeadCT

---

## Project Structure

```
├── data/
│   ├── raw/          ← original Medical MNIST images (one sub-folder per region)
│   └── processed/    ← reserved for any pre-processed artefacts
├── models/           ← saved model weights (.h5)
├── notebooks/        ← DSAI_490_Assignment1_AE_VAE.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py   ← tf.data pipeline helpers
│   ├── model.py             ← build_ae() / build_vae() / VAEModel
│   └── train.py             ← training loops, callbacks
├── tests/
│   ├── test_data_processing.py
│   └── test_model.py
├── README.md
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place your `kaggle.json` in `~/.kaggle/` then run the notebook.  
**Important:** use the original image files — do **not** use any .npz or .csv version.

## Code Conventions

| Rule | Detail |
|------|--------|
| Naming | `snake_case` for variables/functions, `UPPER_SNAKE_CASE` for constants, `PascalCase` for classes |
| Docstrings | Every public function has a one-line summary + `Args:` / `Returns:` block |
| Data loading | `tf.data` exclusively — no manual NumPy loops over datasets |
| Constants | All hyper-parameters defined once at the top of each module |
| Modularity | Logic split across `data_processing`, `model`, `train` — notebook imports from `src/` |

## Running

Open `notebooks/DSAI_490_Assignment1_AE_VAE.ipynb` in Google Colab (GPU runtime recommended).
=======
# Gans_Assignment_1
>>>>>>> 64320ecbf4d4255745b39f11428e200992627078
