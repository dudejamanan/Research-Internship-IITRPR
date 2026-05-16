import os
import cv2
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(preds, masks):
    preds = torch.argmax(preds, dim=1)

    preds = preds.view(-1)
    masks = masks.view(-1)

    TP = ((preds == 1) & (masks == 1)).sum().item()
    FP = ((preds == 1) & (masks == 0)).sum().item()
    FN = ((preds == 0) & (masks == 1)).sum().item()
    TN = ((preds == 0) & (masks == 0)).sum().item()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    prec = TP / (TP + FP + 1e-6)
    rec = TP / (TP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)

    return acc, prec, rec, iou, dice

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    total = {"Accuracy":0,"Precision":0,"Recall":0,"IoU":0,"Dice":0}
    count = 0

    with torch.no_grad():
        for rgb, ms, masks in loader:
            rgb, ms, masks = rgb.to(device), ms.to(device), masks.to(device)

            preds = model(rgb, ms)
            preds = F.interpolate(preds, size=masks.shape[1:], mode='bilinear', align_corners=False)

            acc, prec, rec, iou, dice = compute_metrics(preds, masks)

            for k,v in zip(total.keys(), [acc,prec,rec,iou,dice]):
                total[k] += v

            count += 1

    return {k: v/count for k,v in total.items()}

# -----------------------------
# Dataset
# -----------------------------
class WeedyRiceDataset(Dataset):
    def __init__(self, root_dir, split_file):
        self.rgb_dir = os.path.join(root_dir, "RGB")
        self.ms_dir = os.path.join(root_dir, "Multispectral")
        self.mask_dir = os.path.join(root_dir, "Masks")

        with open(split_file, "r") as f:
            self.samples = [x.strip().replace(".JPG","").replace(".jpg","") for x in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base = self.samples[idx]

        # RGB
        rgb = cv2.imread(os.path.join(self.rgb_dir, base+".JPG"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256,256))
        rgb = rgb.astype(np.float32)/255.0
        rgb = np.transpose(rgb,(2,0,1))

        # Multispectral
        bands = []
        for b in ["_G.TIF","_R.TIF","_RE.TIF","_NIR.TIF"]:
            img = cv2.imread(os.path.join(self.ms_dir, base+b),0)
            img = cv2.resize(img,(256,256))
            img = img.astype(np.float32)/255.0
            bands.append(img)

        ms = np.stack(bands,axis=0)

        # Mask
        mask = cv2.imread(os.path.join(self.mask_dir, base+".png"),0)
        mask = cv2.resize(mask,(256,256))
        mask = (mask > 0).astype(np.int64)

        return (
            torch.tensor(rgb, dtype=torch.float32),
            torch.tensor(ms, dtype=torch.float32),
            torch.tensor(mask)
        )

# -----------------------------
# CRF-like Refinement
# -----------------------------
class CRFRefinement(nn.Module):
    def __init__(self, num_classes=2, iterations=5):
        super().__init__()
        self.iterations = iterations
        self.conv = nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=False)

    def forward(self, x):
        # DO NOT apply softmax here

        for _ in range(self.iterations):
            pairwise = self.conv(x)
            x = x + pairwise   # keep logits

        return x   # return logits

# -----------------------------
# Model
# -----------------------------
class CRF_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

        self.ms_net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=4,
            classes=2
        )

        self.crf = CRFRefinement(2)

    def forward(self, rgb, ms):
        out_rgb = self.rgb_net(rgb)
        out_ms = self.ms_net(ms)

        out = out_rgb + out_ms
        out = self.crf(out)   # now returns logits

        return out

# -----------------------------
# MAIN (5 RUNS)
# -----------------------------
if __name__ == "__main__":

    root = "D:/IIT_Ropar/Datasets/Agriculture/WeedyRice-RGBMS-DB"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_RUNS = 5
    all_results = []

    for run in range(NUM_RUNS):
        print(f"\n===== RUN {run+1} =====")

        set_seed(42 + run)

        train_loader = DataLoader(WeedyRiceDataset(root,"train_list.txt"), batch_size=2, shuffle=True)
        val_loader   = DataLoader(WeedyRiceDataset(root,"val_list.txt"), batch_size=2)
        test_loader  = DataLoader(WeedyRiceDataset(root,"test_list.txt"), batch_size=2)

        model = CRF_UNet().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        ce_loss = nn.CrossEntropyLoss()

        best_iou = 0
        best_model = None

        for epoch in range(20):
            model.train()

            for rgb, ms, masks in train_loader:
                rgb, ms, masks = rgb.to(device), ms.to(device), masks.to(device)

                preds = model(rgb, ms)
                loss = ce_loss(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} | Val IoU: {val['IoU']:.4f}")

            if val["IoU"] > best_iou:
                best_iou = val["IoU"]
                best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # TEST
        model.load_state_dict(best_model)
        model.to(device)

        test = evaluate(model, test_loader, device)
        print("Test:", test)

        all_results.append(test)

    # -----------------------------
    # FINAL STATS
    # -----------------------------
    print("\n===== FINAL RESULTS =====")

    for metric in all_results[0].keys():
        vals = [r[metric] for r in all_results]
        print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # -----------------------------
    # SAVE CSV
    # -----------------------------
    with open("crf_unet_5runs_results.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run","Accuracy","Precision","Recall","IoU","Dice"])

        for i, r in enumerate(all_results):
            writer.writerow([i+1, r["Accuracy"], r["Precision"], r["Recall"], r["IoU"], r["Dice"]])