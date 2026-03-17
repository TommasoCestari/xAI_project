"""
Compute Integrated Gradients attributions for a trained SENN model on FashionMNIST.

Saves:
  - ig_attributions.pt       : full test-set attributions (N, 1, 28, 28)
  - ig_predictions.pt        : model predictions per sample  (N,)
  - ig_labels.pt             : ground-truth labels            (N,)
  - ig_ablation_drops.npy    : per-sample confidence drop after masking top-20% pixels
  - ig_meta.json             : timing + hyperparams

Usage:
    python run_integrated_gradients.py --config configs/fashion_mnist_lambda1e-2_c5_seed29.json
    python run_integrated_gradients.py --config configs/fashion_mnist_lambda1e-2_c5_seed29.json --max_images 500
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from senn.trainer import SENN_Trainer
from captum.attr import IntegratedGradients


# ── helpers ──────────────────────────────────────────────────────────────────

def load_senn(config_path, device="cpu"):
    with open(config_path, "r") as f:
        config = json.load(f)
    config["device"] = device
    config["train"] = False
    config = SimpleNamespace(**config)

    ckpt_path = Path("results") / config.exp_name / "checkpoints" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    trainer = SENN_Trainer(config)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model_state"])
    trainer.model.eval()
    print(f"[IG] Model loaded — best valid acc: {ckpt['best_accuracy']*100:.2f}%")
    return trainer


class SENNWrapper(nn.Module):
    """Expose only the log-softmax output for Captum."""
    def __init__(self, senn_model):
        super().__init__()
        self.senn = senn_model

    def forward(self, x):
        predictions, _, _ = self.senn(x)
        return predictions


def pixel_ablation_confidence_drop(wrapper, images, attributions, pred_labels,
                                   top_fraction=0.20):
    """Mask top-k% pixels (by |attribution|) with the background value; return per-sample confidence drop."""
    
    # Valore corretto del nero (sfondo) dopo la normalizzazione di FashionMNIST
    fill_value = -0.8102 
    
    wrapper.eval()
    with torch.no_grad():
        probs_orig = torch.softmax(wrapper(images), dim=1)
        conf_orig = probs_orig[torch.arange(len(pred_labels)), pred_labels]

        images_abl = images.clone()
        for i in range(len(images)):
            attr_flat = attributions[i].sum(dim=0).abs().flatten()
            k = int(top_fraction * len(attr_flat))
            topk_idx = attr_flat.topk(k).indices
            img_flat = images_abl[i].view(images.shape[1], -1)
            img_flat[:, topk_idx] = fill_value # Applica il background nero

        probs_abl = torch.softmax(wrapper(images_abl), dim=1)
        conf_abl = probs_abl[torch.arange(len(pred_labels)), pred_labels]

    return (conf_orig - conf_abl).cpu().numpy()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Integrated Gradients on a SENN model")
    parser.add_argument("--config", required=True, help="Path to SENN config JSON")
    parser.add_argument("--n_steps", type=int, default=50, help="IG integration steps")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Max test images to process (0 = all)")
    parser.add_argument("--device", default="", help="Device (auto-detect if empty)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[IG] Device: {device}")

    # Load model
    trainer = load_senn(args.config, device=device)
    model = trainer.model
    test_loader = trainer.test_loader
    wrapper = SENNWrapper(model).to(device)
    wrapper.eval()

    ig = IntegratedGradients(wrapper)

    # Output dir
    with open(args.config, "r") as f:
        exp_name = json.load(f)["exp_name"]
    out_dir = Path("results") / exp_name / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect attributions batch by batch
    all_attrs, all_preds, all_labels, all_drops = [], [], [], []
    n_processed = 0
    t_total = 0.0

    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.float().to(device)
        y = y.long().to(device)

        with torch.no_grad():
            preds = model(x)[0].argmax(1)

        baseline = torch.full_like(x, -0.8102) #partiamo da tutto nero
        t0 = time.perf_counter()
        attrs = ig.attribute(x, baselines=baseline, target=preds, n_steps=args.n_steps)
        t_total += time.perf_counter() - t0

        drops = pixel_ablation_confidence_drop(wrapper, x, attrs, preds)

        all_attrs.append(attrs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_drops.append(drops)

        n_processed += len(x)
        print(f"  Batch {batch_idx+1}: {n_processed} samples done "
              f"({t_total:.1f}s elapsed)")

        if args.max_images > 0 and n_processed >= args.max_images:
            break

    all_attrs  = torch.cat(all_attrs)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_drops  = np.concatenate(all_drops)

    # Save
    torch.save(all_attrs,  out_dir / "ig_attributions.pt")
    torch.save(all_preds,  out_dir / "ig_predictions.pt")
    torch.save(all_labels, out_dir / "ig_labels.pt")
    np.save(out_dir / "ig_ablation_drops.npy", all_drops)

    meta = {
        "method": "IntegratedGradients",
        "config": args.config,
        "exp_name": exp_name,
        "n_steps": args.n_steps,
        "n_samples": int(len(all_labels)),
        "total_time_s": round(t_total, 3),
        "time_per_sample_s": round(t_total / len(all_labels), 5),
        "mean_confidence_drop": round(float(all_drops.mean()), 6),
        "std_confidence_drop": round(float(all_drops.std()), 6),
        "device": device,
    }
    with open(out_dir / "ig_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[IG] Done — {len(all_labels)} samples in {t_total:.1f}s")
    print(f"     Mean confidence drop (top-20% ablation): {all_drops.mean():.4f}")
    print(f"     Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
