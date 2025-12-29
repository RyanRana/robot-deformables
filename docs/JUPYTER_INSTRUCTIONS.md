# Running Step 3 via JupyterLab (Easiest Method!)

## Step 1: Access JupyterLab

1. Go to your Vast.ai instances page
2. Click **"Connect"** on your instance
3. Click **"Open JupyterLab"** (or go to http://171.250.15.13:8080)

## Step 2: Upload Files

In JupyterLab:
1. Click the **upload** button (↑ icon) in the file browser
2. Upload ALL files from `collobarative robot` folder:
   - data_preparation.py
   - deformable_handover_env.py
   - diffusion_policy.py
   - train_step3.py
   - requirements.txt
   - (and all other .py files)

## Step 3: Open Terminal

In JupyterLab:
1. Click **"File"** → **"New"** → **"Terminal"**

## Step 4: Install Dependencies

In the terminal, paste:

```bash
pip install datasets numpy pillow huggingface-hub gymnasium
```

## Step 5: Verify GPU

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Step 6: Start Training

```bash
python3 train_step3.py
```

## Done!

Training will run and show progress. Takes about 1.5-2 hours.

You can close the browser - training will continue!

---

## To Check Progress Later

1. Open JupyterLab again
2. Open Terminal
3. Check if training is still running:
   ```bash
   ps aux | grep python
   ```
4. View logs:
   ```bash
   ls -lh
   # If training.log exists:
   tail -f training.log
   ```

