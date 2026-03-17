import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from senn.trainer import SENN_Trainer

# ── helpers ──────────────────────────────────────────────────────────────────

def load_senn(config_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)
    config["device"] = device
    config["train"] = False
    config = SimpleNamespace(**config)

    ckpt_path = Path("results") / config.exp_name / "checkpoints" / "best_model.pt"
    trainer = SENN_Trainer(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.model.eval()
    return trainer

class SENNWrapper(nn.Module):
    def __init__(self, senn_model):
        super().__init__()
        self.senn = senn_model

    def forward(self, x):
        predictions, _, _ = self.senn(x)
        return predictions

def compute_top_vs_random_drops(wrapper, images, attributions, pred_labels, top_fraction=0.20):
    """Compute both Top-K% and Random-K% ablation confidence drops."""
    fill_value = -0.8102  # Valore di background (nero normalizzato)
    wrapper.eval()
    
    with torch.no_grad():
        # Confidenza originale
        probs_orig = torch.softmax(wrapper(images), dim=1)
        conf_orig = probs_orig[torch.arange(len(pred_labels)), pred_labels]

        images_top = images.clone()
        images_rand = images.clone()
        
        k = int(top_fraction * images.shape[2] * images.shape[3]) # Es. 20% di 784 = 156 pixel

        for i in range(len(images)):
            # 1. Maschera i Top pixel (trovati da LIME)
            attr_flat = attributions[i].sum(dim=0).abs().flatten()
            topk_idx = attr_flat.topk(k).indices
            images_top[i].view(images.shape[1], -1)[:, topk_idx] = fill_value
            
            # 2. Maschera K pixel Casuali (Random baseline)
            rand_idx = torch.randperm(images.shape[2] * images.shape[3])[:k]
            images_rand[i].view(images.shape[1], -1)[:, rand_idx] = fill_value

        # Ricalcola la confidenza
        conf_top = torch.softmax(wrapper(images_top), dim=1)[torch.arange(len(pred_labels)), pred_labels]
        conf_rand = torch.softmax(wrapper(images_rand), dim=1)[torch.arange(len(pred_labels)), pred_labels]

    drop_top = (conf_orig - conf_top).cpu().numpy()
    drop_rand = (conf_orig - conf_rand).cpu().numpy()
    
    return drop_top, drop_rand

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute Relative Metrics from saved LIME run")
    parser.add_argument("--config", required=True, help="Path to SENN config JSON")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Carica le impostazioni e trova la cartella dei risultati
    with open(args.config, "r") as f:
        exp_name = json.load(f)["exp_name"]
    out_dir = Path("results") / exp_name / "posthoc"
    
    # 2. Carica i risultati salvati da LIME
    print("Caricamento risultati LIME salvati...")
    lime_attrs = torch.load(out_dir / "lime_attributions.pt", map_location=device)
    lime_preds = torch.load(out_dir / "lime_predictions.pt", map_location=device)
    
    n_samples = len(lime_attrs)
    print(f"Trovate {n_samples} spiegazioni pre-calcolate.")

    # 3. Carica il modello e il DataLoader
    trainer = load_senn(args.config, device=device)
    wrapper = SENNWrapper(trainer.model).to(device)
    
    # Raccogli le immagini originali corrispondenti
    # (Essendo shuffle=False nel test_loader, l'ordine è deterministico)
    all_images = []
    collected = 0
    for x, _ in trainer.test_loader:
        all_images.append(x)
        collected += len(x)
        if collected >= n_samples:
            break
            
    all_images = torch.cat(all_images)[:n_samples].to(device)

    # 4. Calcola i drop a lotti (per non riempire la memoria GPU)
    batch_size = 128
    all_drop_top, all_drop_rand = [], []
    
    print("Calcolo Metrica Relativa (Top vs Random)...")
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batch_x = all_images[i:end]
        batch_attr = lime_attrs[i:end]
        batch_preds = lime_preds[i:end]
        
        d_top, d_rand = compute_top_vs_random_drops(wrapper, batch_x, batch_attr, batch_preds)
        all_drop_top.extend(d_top)
        all_drop_rand.extend(d_rand)

    all_drop_top = np.array(all_drop_top)
    all_drop_rand = np.array(all_drop_rand)
    relative_drops = all_drop_top - all_drop_rand
    
    # 5. Salva i nuovi risultati e stampa il verdetto
    np.save(out_dir / "lime_drop_top.npy", all_drop_top)
    np.save(out_dir / "lime_drop_rand.npy", all_drop_rand)
    np.save(out_dir / "lime_drop_relative.npy", relative_drops)
    
    print("\n" + "="*50)
    print("RISULTATI FINALI (Mean Confidence Drop):")
    print("="*50)
    print(f"Ablation Top-20% Pixel   : {all_drop_top.mean():.4f}")
    print(f"Ablation Random-20% Pixel: {all_drop_rand.mean():.4f}")
    print(f"VANTAGGIO LIME (Top-Rand): {relative_drops.mean():.4f}")
    print("="*50)
    print(f"I file numpy aggiornati sono stati salvati in {out_dir}")

if __name__ == "__main__":
    main()