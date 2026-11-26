# Acrobot-v1: Fine-Tuning a Pretrained PPO Agent

Fine-tuning the pretrained Stable-Baselines3 PPO agent for Gymnasium's **Acrobot-v1** (reward closer to 0 is better), with a safe early-stopping callback and a small FastAPI web app to compare the agents and watch the result.

**Stack:** Python, Stable-Baselines3, Gymnasium, FastAPI, imageio

## Results (20 evaluation episodes, deterministic policy)
| Model | Mean reward |
|---|---|
| Pretrained (`sb3/ppo-Acrobot-v1`) | -89.35 +/- 35.99 |
| Fine-tuned (best checkpoint) | **-80.75 +/- 23.52** |
| Difference | +8.60 |

The fine-tuned agent also has lower variance across episodes. The gain is modest and measured on 20 episodes, so treat it as an improvement in consistency more than a large jump in reward.

## What I did
1. Loaded the pretrained PPO agent from the Hugging Face Hub and measured a baseline.
2. Inspected its hyperparameters and network (64x64 MLP policy and value nets).
3. Fine-tuned with conservative settings: learning rate 1e-5, clip range 0.05, n_steps 1024, batch size 256, entropy coefficient 0.0005, 4 parallel envs.
4. Wrote a custom `EarlyStoppingBestWeights` callback that evaluates periodically, saves the best checkpoint, and stops on a reward floor or a sharp regression, so a bad update cannot overwrite a good model.
5. Reloaded the best checkpoint (not the last one) for the final evaluation.
6. Built a FastAPI app (`main.py`) that compares both models and renders the fine-tuned agent as a GIF.

## Files
| File | Purpose |
|---|---|
| `Acrobot-v1-PPO_Fixed.ipynb` | Baseline, fine-tuning, early stopping, evaluation |
| `main.py` | FastAPI app: model comparison and GIF playback |
| `ppo-Acrobot-v1.zip` | Pretrained model |
| `best_acrobot_model.zip` | Best fine-tuned checkpoint |
| `requirements.txt` | Dependencies |

## Run the app
```bash
pip install -r requirements.txt
python main.py     # then open http://localhost:8000
```
