import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5EncoderModel, EsmModel, EsmTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, confusion_matrix
import warnings
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import glob

warnings.filterwarnings("ignore")
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache/transformers"

# ==================== Configuration ====================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 40
LEARNING_RATE = 1e-4
PATIENCE = 8
KFOLD_NUM = 5
DATA_FOLDER = "/root/autodl-tmp/So4/xun"
PROT_T5_MODEL = "/root/autodl-tmp/models/prot_t5_xl_uniref50"
ESM_MODEL_NAME = "/root/autodl-tmp/esm2_t30_150M_UR50D"
MODEL_SAVE_DIR = "/root/autodl-tmp/trained_models"
MANUAL_POS_WEIGHT = 220

PLOT_SAVE_DIR = "/root/autodl-tmp/plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("training_log.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("protein_classifier")

NEG_COLOR = '#DAB1DA'
POS_COLOR = '#81D8D0'
ALPHA = 0.6
SIZE = 40

MAX_SEQ_LEN = 1022

# ==================== Helper Functions ====================
def format_seq(seq):
    return ' '.join(seq)

# ==================== Data Loading ====================
def load_data_from_folder(folder_path):
    seqs = []
    labels = []
    ids = []
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found. Please check folder path: {folder_path}")
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 3:
            logger.warning(f"Skipping {file_path}: fewer than 3 lines")
            continue
        seq_id = lines[0].lstrip('>')
        sequence = lines[1]
        label_str = lines[2]
        label = [int(c) for c in label_str if c in '01']
        if len(label) != len(sequence):
            logger.warning(f"Skipping {file_path}: sequence length {len(sequence)} does not match label length {len(label)}")
            continue
        seqs.append(sequence)
        labels.append(label)
        ids.append(seq_id)
    logger.info(f"Successfully loaded {len(seqs)} samples")
    lengths = [len(s) for s in seqs]
    logger.info(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    pos_counts = [sum(l) for l in labels]
    total_pos = sum(pos_counts)
    total_res = sum(lengths)
    logger.info(f"Positive sample ratio: {total_pos}/{total_res} = {total_pos / total_res:.4f}")
    return seqs, labels, ids

# ==================== Dataset ====================
class ProteinResidueDataset(Dataset):
    def __init__(self, seqs, labels, prot5_tokenizer):
        self.seqs = seqs
        self.labels = labels
        self.prot5_tokenizer = prot5_tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        seq_fmt = format_seq(seq)
        encoding = self.prot5_tokenizer(
            seq_fmt,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=MAX_SEQ_LEN + 2
        )
        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        valid_len = attention_mask.sum().item() - 2
        if valid_len < 0:
            valid_len = 0
        label = label[:valid_len]

        return {
            "seq_str": seq,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "seq_len": valid_len
        }

def collate_fn(batch):
    seq_strs = [item["seq_str"] for item in batch]
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["label"] for item in batch]
    seq_lens = [item["seq_len"] for item in batch]

    max_len = max(ids.size(0) for ids in input_ids)
    padded_input_ids = []
    padded_attn_mask = []
    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        else:
            padded_ids = ids
            padded_mask = mask
        padded_input_ids.append(padded_ids)
        padded_attn_mask.append(padded_mask)

    return {
        "seq_strs": seq_strs,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attn_mask),
        "labels": labels,
        "seq_lens": seq_lens
    }

# ==================== CNN Module ====================
class CNNBlock(nn.Module):
    def __init__(self, hidden_size, out_channels, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        identity = x
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        x = x + identity
        x = self.relu(x)
        return x

# ==================== Model: Cross-Attention Fusion + CNN + Self-Attention ====================
class ProtT5ESMWithCNN(nn.Module):
    def __init__(self, prot5_model, esm_model, prot5_dim, esm_dim,
                 d_model=256, nhead=8, cnn_channels=256, dropout=0.3):
        super().__init__()
        self.prot5 = prot5_model
        self.esm = esm_model
        self.prot5_dim = prot5_dim
        self.esm_dim = esm_dim
        self.cnn_channels = cnn_channels
        self.d_model = d_model

        for param in self.prot5.parameters():
            param.requires_grad = False
        for param in self.esm.parameters():
            param.requires_grad = False

        self.prot5_proj = nn.Linear(prot5_dim, d_model)
        self.esm_proj = nn.Linear(esm_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.cnn = CNNBlock(d_model, cnn_channels, dropout_p=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=cnn_channels, num_heads=nhead, batch_first=True)
        self.residue_classifier = nn.Linear(cnn_channels, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.esm_tokenizer = None

    def forward(self, input_ids, attention_mask, seq_strs, return_features=False):
        prot5_out = self.prot5(input_ids=input_ids, attention_mask=attention_mask)
        prot5_emb = prot5_out.last_hidden_state

        device = prot5_emb.device
        batch_size = len(seq_strs)
        esm_emb_list = []
        with torch.no_grad():
            for seq in seq_strs:
                trunc_seq = seq[:MAX_SEQ_LEN]
                enc = self.esm_tokenizer(trunc_seq, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN+2)
                enc = {k: v.to(device) for k, v in enc.items()}
                esm_out = self.esm(**enc)
                esm_emb = esm_out.last_hidden_state[0, 1:-1, :]
                esm_emb_list.append(esm_emb)

        all_logits = []
        all_cnn_feats = []
        all_attn_feats = []
        for b in range(batch_size):
            valid_len = attention_mask[b].sum().item() - 2
            if valid_len <= 0:
                all_logits.append(torch.zeros(0, device=device))
                all_cnn_feats.append(torch.zeros(0, self.cnn_channels, device=device))
                all_attn_feats.append(torch.zeros(0, self.cnn_channels, device=device))
                continue

            prot5_feat = prot5_emb[b, 1:1+valid_len, :]
            esm_feat = esm_emb_list[b]
            min_len = min(prot5_feat.shape[0], esm_feat.shape[0])
            prot5_feat = prot5_feat[:min_len, :]
            esm_feat = esm_feat[:min_len, :]

            prot5_proj = self.prot5_proj(prot5_feat)
            esm_proj = self.esm_proj(esm_feat)

            attn_out, _ = self.cross_attn(prot5_proj.unsqueeze(0), esm_proj.unsqueeze(0), esm_proj.unsqueeze(0))
            fused = self.norm(prot5_proj + attn_out.squeeze(0))

            fused = fused.unsqueeze(0).transpose(1, 2)
            cnn_out = self.cnn(fused)
            cnn_feat = cnn_out.transpose(1, 2)

            attn_feat, _ = self.attention(cnn_feat, cnn_feat, cnn_feat)
            attn_feat = self.dropout(attn_feat)

            logits = self.residue_classifier(attn_feat).squeeze(-1)
            logits = logits.squeeze(0)

            all_logits.append(logits)
            if return_features:
                all_cnn_feats.append(cnn_feat.squeeze(0))
                all_attn_feats.append(attn_feat.squeeze(0))

        max_len = max(l.shape[0] for l in all_logits)
        padded_logits = torch.zeros(batch_size, max_len, device=device)
        for b, log in enumerate(all_logits):
            padded_logits[b, :log.shape[0]] = log

        if return_features:
            max_feat_len = max(f.shape[0] for f in all_cnn_feats)
            padded_cnn = torch.zeros(batch_size, max_feat_len, self.cnn_channels, device=device)
            padded_attn = torch.zeros(batch_size, max_feat_len, self.cnn_channels, device=device)
            for b, (c, a) in enumerate(zip(all_cnn_feats, all_attn_feats)):
                padded_cnn[b, :c.shape[0], :] = c
                padded_attn[b, :a.shape[0], :] = a
            return padded_logits, padded_cnn, padded_attn
        return padded_logits

def build_model(prot5_model_path, esm_model_name, esm_tokenizer):
    prot5 = T5EncoderModel.from_pretrained(prot5_model_path)
    esm = EsmModel.from_pretrained(esm_model_name)
    prot5_dim = prot5.config.hidden_size
    esm_dim = esm.config.hidden_size
    model = ProtT5ESMWithCNN(prot5, esm, prot5_dim, esm_dim,
                             d_model=256, nhead=8, cnn_channels=256, dropout=0.2)
    model.esm_tokenizer = esm_tokenizer
    return model.to(DEVICE)

# ==================== Loss Functions ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight)
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class ProxyNCALoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=256, temperature=0.1):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.temperature = temperature

    def forward(self, features, labels):
        dist = torch.cdist(features, self.proxies, p=2) ** 2
        sim = -dist / self.temperature
        log_probs = nn.functional.log_softmax(sim, dim=1)
        loss = -log_probs[range(len(labels)), labels].mean()
        return loss

# ==================== t-SNE Visualization ====================
def plot_residue_tsne(features, labels, title, filename, max_points=20000):
    if features is None or features.shape[0] == 0:
        logger.warning(f"Skipping plot {filename}: no valid features")
        return
    if features.shape[0] > max_points:
        idx = np.random.choice(features.shape[0], max_points, replace=False)
        features = features[idx]
        labels = labels[idx]
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(features)
    if feat_scaled.shape[1] > 50:
        pca = PCA(n_components=min(50, feat_scaled.shape[0]), random_state=42)
        feat_pca = pca.fit_transform(feat_scaled)
    else:
        feat_pca = feat_scaled
    perplexity = min(30, max(5, feat_pca.shape[0] // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
    feat_2d = tsne.fit_transform(feat_pca)
    plt.figure(figsize=(8, 6), dpi=150)
    ax = plt.gca()
    if 0 in labels:
        ax.scatter(feat_2d[labels == 0, 0], feat_2d[labels == 0, 1],
                   c=NEG_COLOR, label='Non-binding', alpha=ALPHA, s=SIZE, edgecolors='none')
    if 1 in labels:
        ax.scatter(feat_2d[labels == 1, 0], feat_2d[labels == 1, 1],
                   c=POS_COLOR, label='Binding', alpha=ALPHA, s=SIZE, edgecolors='none')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_SAVE_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close('all')
    logger.info(f"Saved t-SNE plot: {filename} (sampled {features.shape[0]} residues)")

# ==================== Evaluation ====================
def compute_metrics(y_true, y_pred_proba):
    y_pred = (y_pred_proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    sn = tp / (tp + fn) if (tp + fn) != 0 else 0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
    return {"ACC": acc, "SN": sn, "SP": sp, "MCC": mcc, "AUC": auc}

# ==================== Single-Fold Training ====================
def train_fold(model, train_loader, val_loader, focal_loss, proxy_loss, optimizer, lambda_proxy=0.2, fold=0):
    best_mcc = -1.0
    patience = 0
    best_metric = None
    best_cnn_feats = None
    best_attn_feats = None
    best_labels = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            seq_strs = batch["seq_strs"]
            labels_list = batch["labels"]
            seq_lens = batch["seq_lens"]

            residue_logits, cnn_features, attn_features = model(input_ids, attn_mask, seq_strs, return_features=True)

            batch_focal_loss = 0
            batch_proxy_loss = 0
            for b in range(len(labels_list)):
                L = seq_lens[b]
                if L <= 0:
                    continue
                logits_b = residue_logits[b, :L]
                feat_b = attn_features[b, :L, :]
                label_b = labels_list[b].to(DEVICE)
                min_len = min(logits_b.shape[0], label_b.shape[0])
                logits_b = logits_b[:min_len]
                feat_b = feat_b[:min_len, :]
                label_b = label_b[:min_len]
                if min_len == 0:
                    continue
                batch_focal_loss += focal_loss(logits_b, label_b)
                batch_proxy_loss += proxy_loss(feat_b, label_b.long())
            loss = (batch_focal_loss / len(labels_list)) + lambda_proxy * (batch_proxy_loss / len(labels_list))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_y_true = []
        all_y_prob = []
        all_cnn_residue_feats = []
        all_attn_residue_feats = []
        all_residue_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                seq_strs = batch["seq_strs"]
                labels_list = batch["labels"]
                seq_lens = batch["seq_lens"]
                residue_logits, cnn_features, attn_features = model(input_ids, attn_mask, seq_strs, return_features=True)
                for b in range(len(labels_list)):
                    L = seq_lens[b]
                    if L <= 0:
                        continue
                    logits_b = residue_logits[b, :L]
                    probs = torch.sigmoid(logits_b).cpu().numpy()
                    label_b = labels_list[b].numpy()[:L]
                    all_y_true.extend(label_b)
                    all_y_prob.extend(probs)
                    cnn_b = cnn_features[b, :L, :].cpu().numpy()
                    attn_b = attn_features[b, :L, :].cpu().numpy()
                    all_cnn_residue_feats.append(cnn_b)
                    all_attn_residue_feats.append(attn_b)
                    all_residue_labels.append(label_b)
        if all_cnn_residue_feats:
            all_cnn_residue_feats = np.vstack(all_cnn_residue_feats)
            all_attn_residue_feats = np.vstack(all_attn_residue_feats)
            all_residue_labels = np.concatenate(all_residue_labels)
        else:
            all_cnn_residue_feats = np.array([])
            all_attn_residue_feats = np.array([])
            all_residue_labels = np.array([])

        metrics = compute_metrics(np.array(all_y_true), np.array(all_y_prob))
        logger.info(f"Fold{fold} Epoch{epoch + 1} | TrainLoss {train_loss / len(train_loader):.4f} | "
                    f"ACC {metrics['ACC']:.4f} SN {metrics['SN']:.4f} SP {metrics['SP']:.4f} "
                    f"MCC {metrics['MCC']:.4f} AUC {metrics['AUC']:.4f}")

        if metrics['MCC'] > best_mcc:
            best_mcc = metrics['MCC']
            best_metric = metrics
            patience = 0
            best_cnn_feats = all_cnn_residue_feats.copy() if all_cnn_residue_feats.size > 0 else None
            best_attn_feats = all_attn_residue_feats.copy() if all_attn_residue_feats.size > 0 else None
            best_labels = all_residue_labels.copy() if all_residue_labels.size > 0 else None
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                "fold": fold,
                "model": model.state_dict(),
                "metric": best_metric
            }, os.path.join(MODEL_SAVE_DIR, f"best_fold{fold}.pth"))
        else:
            patience += 1
            if patience >= PATIENCE:
                logger.info("Early stopping")
                break

    if best_cnn_feats is not None and best_labels is not None:
        plot_residue_tsne(best_cnn_feats, best_labels,
                          title=f'Fold {fold} - Best Epoch CNN Features (Cross-Attn)',
                          filename=f"fold{fold}_best_cnn_tsne.png")
        plot_residue_tsne(best_attn_feats, best_labels,
                          title=f'Fold {fold} - Best Epoch Attention Features (Cross-Attn)',
                          filename=f"fold{fold}_best_attn_tsne.png")
    return best_metric

# ==================== Main ====================
if __name__ == "__main__":
    seqs, labels, ids = load_data_from_folder(DATA_FOLDER)
    if len(seqs) == 0:
        raise RuntimeError("No samples loaded successfully. Please check the folder format.")

    prot5_tokenizer = T5Tokenizer.from_pretrained(PROT_T5_MODEL, do_lower_case=False, legacy=False, use_fast=False)
    esm_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)

    # Extract cross-attention fusion features for t-SNE
    logger.info("Extracting ProtT5+ESM cross-attention fusion features for t-SNE...")
    prot5_model = T5EncoderModel.from_pretrained(PROT_T5_MODEL).to(DEVICE).eval()
    esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE).eval()
    for p in prot5_model.parameters():
        p.requires_grad = False
    for p in esm_model.parameters():
        p.requires_grad = False

    d_model = 256
    nhead = 8
    prot5_dim = prot5_model.config.hidden_size
    esm_dim = esm_model.config.hidden_size
    prot5_proj = nn.Linear(prot5_dim, d_model).to(DEVICE).eval()
    esm_proj = nn.Linear(esm_dim, d_model).to(DEVICE).eval()
    cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True).to(DEVICE).eval()
    norm = nn.LayerNorm(d_model).to(DEVICE).eval()

    torch.manual_seed(42)
    for layer in [prot5_proj, esm_proj, cross_attn, norm]:
        for param in layer.parameters():
            if param.requires_grad and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    all_fusion_feats = []
    all_fusion_labels = []
    batch_size_extract = 4

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size_extract):
            batch_seqs = seqs[i:i+batch_size_extract]
            batch_labels = labels[i:i+batch_size_extract]
            batch_seqs_fmt = [format_seq(s) for s in batch_seqs]
            prot5_enc = prot5_tokenizer(batch_seqs_fmt, return_tensors='pt', padding=True,
                                        truncation=True, max_length=MAX_SEQ_LEN+2).to(DEVICE)
            prot5_out = prot5_model(**prot5_enc)
            for b in range(len(batch_seqs)):
                seq = batch_seqs[b]
                label = np.array(batch_labels[b])
                prot5_valid_len = prot5_enc.attention_mask[b].sum().item() - 2
                if prot5_valid_len <= 0:
                    continue
                prot5_feat = prot5_out.last_hidden_state[b, 1:1+prot5_valid_len, :]
                esm_enc = esm_tokenizer(seq, return_tensors='pt', truncation=True, max_length=MAX_SEQ_LEN+2).to(DEVICE)
                esm_out = esm_model(**esm_enc)
                esm_valid_len = esm_enc.attention_mask[0].sum().item() - 2
                if esm_valid_len <= 0:
                    continue
                esm_feat = esm_out.last_hidden_state[0, 1:1+esm_valid_len, :]

                min_len = min(prot5_feat.shape[0], esm_feat.shape[0], len(label))
                if min_len <= 0:
                    continue
                prot5_feat = prot5_feat[:min_len, :]
                esm_feat = esm_feat[:min_len, :]
                label = label[:min_len]

                prot5_proj_out = prot5_proj(prot5_feat)
                esm_proj_out = esm_proj(esm_feat)

                attn_out, _ = cross_attn(prot5_proj_out.unsqueeze(0), esm_proj_out.unsqueeze(0), esm_proj_out.unsqueeze(0))
                fused = norm(prot5_proj_out + attn_out.squeeze(0))

                fusion_feat = fused.cpu().numpy()
                all_fusion_feats.append(fusion_feat)
                all_fusion_labels.append(label)

    if all_fusion_feats:
        all_fusion_feats = np.vstack(all_fusion_feats)
        all_fusion_labels = np.concatenate(all_fusion_labels)
        plot_residue_tsne(all_fusion_feats, all_fusion_labels,
                          title="ProtT5+ESM Cross-Attention Fusion Features (Residue Level)",
                          filename="raw_prott5_esm_crossattn_fusion_tsne.png")
    else:
        logger.warning("No residue features available for cross-attention fusion t-SNE")

    del prot5_model, esm_model, prot5_proj, esm_proj, cross_attn, norm
    torch.cuda.empty_cache()

    dataset = ProteinResidueDataset(seqs, labels, prot5_tokenizer)
    kfold = KFold(n_splits=KFOLD_NUM, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(seqs)):
        logger.info(f"\n===== Fold {fold + 1}/{KFOLD_NUM} =====")
        train_subsampler = torch.utils.data.Subset(dataset, train_idx)
        val_subsampler = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subsampler, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_subsampler, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)

        model = build_model(PROT_T5_MODEL, ESM_MODEL_NAME, esm_tokenizer)
        pos_weight_val = torch.tensor([MANUAL_POS_WEIGHT], dtype=torch.float32).to(DEVICE)
        logger.info(f"Pos weight: {pos_weight_val.item():.2f}")
        focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, pos_weight=pos_weight_val)
        proxy_loss_fn = ProxyNCALoss(num_classes=2, feat_dim=256, temperature=0.1).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        best = train_fold(model, train_loader, val_loader, focal_loss_fn, proxy_loss_fn,
                          optimizer, lambda_proxy=0, fold=fold + 1)
        if best:
            logger.info(f"Fold {fold + 1} Best: ACC={best['ACC']:.4f} MCC={best['MCC']:.4f} AUC={best['AUC']:.4f}")
            all_metrics.append(best)

    if all_metrics:
        logger.info("\n===== Final Results =====")
        logger.info(f"ACC: {np.mean([m['ACC'] for m in all_metrics]):.4f} ± {np.std([m['ACC'] for m in all_metrics]):.4f}")
        logger.info(f"SN : {np.mean([m['SN'] for m in all_metrics]):.4f} ± {np.std([m['SN'] for m in all_metrics]):.4f}")
        logger.info(f"SP : {np.mean([m['SP'] for m in all_metrics]):.4f} ± {np.std([m['SP'] for m in all_metrics]):.4f}")
        logger.info(f"MCC: {np.mean([m['MCC'] for m in all_metrics]):.4f} ± {np.std([m['MCC'] for m in all_metrics]):.4f}")
        logger.info(f"AUC: {np.mean([m['AUC'] for m in all_metrics]):.4f} ± {np.std([m['AUC'] for m in all_metrics]):.4f}")
    else:
        logger.error("No metrics collected.")