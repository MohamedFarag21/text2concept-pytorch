from pathlib import Path
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from einops import rearrange
from tqdm.auto import tqdm
from accelerate import Accelerator

# ──────────────────────────────────────────────
# constants
# ──────────────────────────────────────────────

# Standard CLIP ImageNet zero-shot templates (Radford et al., 2021)
IMAGENET_TEMPLATES = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a photo of many {}.',
    'a photo of the {}.',
    'a bad photo of a {}.',
    'a rendering of a {}.',
    'a photo of the large {}.',
    'a photo of the small {}.',
    'itap of a {}.',
    'a close-up photo of a {}.',
    'a bright photo of a {}.',
    'a black and white photo of the {}.',
    'a low resolution photo of the {}.',
    'art of a {}.',
    'a tattoo of a {}.',
]

# ──────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def flatten_features(t):
    """Collapse spatial / sequence dims — keeps only (batch, features)."""
    if t.dim() > 2:
        t = rearrange(t, 'b ... -> b (...)')
    return t

def infer_encoder_dim(encoder, image_size = 224):
    """Run a single dummy forward pass to detect the encoder output dimension."""
    device = next(encoder.parameters()).device
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device = device)
        out   = encoder(dummy)
        dim   = flatten_features(out).shape[-1]
    encoder.train(was_training)
    return dim

# ──────────────────────────────────────────────
# dataset
# ──────────────────────────────────────────────

class PairedTransformDataset(Dataset):
    """
    Returns the same image twice: once preprocessed for the source encoder,
    once preprocessed for CLIP.  Recursively searches any folder structure.
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    def __init__(self, folder, transform_source, transform_clip):
        self.transform_source = transform_source
        self.transform_clip   = transform_clip

        self.paths = sorted([
            p for p in Path(folder).rglob('*')
            if p.suffix.lower() in self.EXTENSIONS
        ])
        assert len(self.paths) > 0, f'No images found under {folder}'

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform_source(img), self.transform_clip(img)

# ──────────────────────────────────────────────
# aligner
# ──────────────────────────────────────────────

class Text2ConceptAligner(nn.Module):
    """
    Learned linear mapping: source encoder feature space → CLIP vision space.

    Once trained, aligned features are directly comparable to CLIP text
    embeddings via cosine similarity — no concept labels required.
    """

    def __init__(self, source_dim, clip_dim = 512):
        super().__init__()
        self.linear   = nn.Linear(source_dim, clip_dim, bias = True)
        self.clip_dim = clip_dim

    @property
    def source_dim(self):
        return self.linear.in_features

    def forward(self, x):
        x = flatten_features(x)
        return l2norm(self.linear(x))

    @classmethod
    def from_checkpoint(cls, path, device = 'cpu'):
        """Load a trained aligner directly from a .pt checkpoint."""
        data    = torch.load(path, map_location = device, weights_only = True)
        aligner = cls(data['source_dim'], data['clip_dim'])
        aligner.load_state_dict(data['aligner'])
        return aligner.to(device)

# ──────────────────────────────────────────────
# Text2Concept
# ──────────────────────────────────────────────

class Text2Concept(nn.Module):
    """
    Concept Activation Vectors directly from text, for any vision encoder.

    Extends CLIP's shared text-vision space to an arbitrary off-the-shelf
    model via a trained linear aligner (Text2ConceptAligner).

    Usage
    -----
    >>> aligner = Text2ConceptAligner.from_checkpoint('results/aligner-final.pt')
    >>> t2c     = Text2Concept(encoder, aligner)
    >>>
    >>> # similarity to arbitrary text concepts
    >>> sims = t2c.concept_similarity(images, ['red food', 'in a tree'])
    >>>
    >>> # zero-shot classification
    >>> preds = t2c.zero_shot_classify(images, ['cat', 'dog', 'car'])
    >>>
    >>> # CAV in source encoder space (for TCAV-style analysis)
    >>> cav = t2c.get_cav('stripes')
    """

    def __init__(
        self,
        encoder,
        aligner,
        *,
        clip_model_name  = 'ViT-B-16',
        clip_pretrained  = 'openai',
        templates        = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.aligner = aligner

        import open_clip
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained = clip_pretrained
        )
        self.clip_model      = clip_model.eval()
        self.clip_preprocess = clip_preprocess
        self.tokenizer       = open_clip.get_tokenizer(clip_model_name)

        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.templates = default(templates, IMAGENET_TEMPLATES)

    @property
    def device(self):
        return next(self.aligner.parameters()).device

    # ── concept encoding ──────────────────────────

    @torch.inference_mode()
    def encode_concept(self, concept, class_names = None):
        """
        Encode a text concept as a unit vector in CLIP space.

        Args
        ----
        concept     : str  — e.g. 'red food', 'in a tree', 'stripes'
        class_names : list[str], optional — if given, refines the vector by
                      averaging 'a photo of a {cls} {concept}' over all classes,
                      producing a more class-aware concept direction.

        Returns
        -------
        Tensor (1, clip_dim), L2-normalised
        """
        if exists(class_names):
            texts = [f'a photo of a {cls} {concept}' for cls in class_names]
        else:
            texts = [t.format(concept) for t in self.templates]

        tokens     = self.tokenizer(texts).to(self.device)
        embeddings = self.clip_model.encode_text(tokens)
        embeddings = l2norm(embeddings)
        return l2norm(embeddings.mean(dim = 0, keepdim = True))

    # ── image encoding ────────────────────────────

    @torch.inference_mode()
    def encode_images(self, images):
        """
        Encode images: source encoder → aligner → CLIP space.

        Args
        ----
        images : Tensor (b, 3, H, W), preprocessed for the source encoder

        Returns
        -------
        Tensor (b, clip_dim), L2-normalised
        """
        self.encoder.eval()
        features = self.encoder(images.to(self.device))
        return self.aligner(features)

    # ── applications ──────────────────────────────

    def concept_similarity(self, images, concepts):
        """
        Cosine similarity of image features to a list of text concepts.

        Args
        ----
        images   : Tensor (b, 3, H, W)
        concepts : list[str]

        Returns
        -------
        Tensor (b, num_concepts) — values in [-1, 1]
        """
        img_feats = self.encode_images(images)
        con_feats = torch.cat([self.encode_concept(c) for c in concepts], dim = 0)
        return einsum('b d, c d -> b c', img_feats, con_feats)

    def zero_shot_classify(self, images, class_names):
        """
        Zero-shot classification: treats each class name as a concept,
        returns the index of the most similar class.

        Args
        ----
        images      : Tensor (b, 3, H, W)
        class_names : list[str]

        Returns
        -------
        Tensor (b,) — predicted class indices
        """
        return self.concept_similarity(images, class_names).argmax(dim = -1)

    def get_cav(self, concept, class_names = None):
        """
        Concept Activation Vector in the **source encoder's** feature space.

        Obtained by projecting the CLIP text embedding back through W^T,
        giving a direction in source space that points toward the concept.
        Useful for TCAV-style sensitivity analysis.

        Returns
        -------
        Tensor (1, source_dim), L2-normalised
        """
        concept_vec = self.encode_concept(concept, class_names)  # (1, clip_dim)
        W           = self.aligner.linear.weight                 # (clip_dim, source_dim)
        cav         = concept_vec @ W                            # (1, source_dim)
        return l2norm(cav)

# ──────────────────────────────────────────────
# trainer
# ──────────────────────────────────────────────

class Trainer:
    """
    Trains a Text2ConceptAligner on paired image representations.

    For each image in the folder, extracts features from both the source
    encoder and CLIP's vision encoder, then minimises cosine distance
    between the aligned source features and the CLIP features.

    Args
    ----
    encoder          : nn.Module — any frozen vision encoder (ResNet, ViT, DINO…)
    folder           : str / Path — directory of training images (ImageNet recommended)
    source_transform : torchvision transform for the source encoder.
                       Defaults to CLIP's own val transform if not provided.
    clip_model_name  : open_clip model identifier (default 'ViT-B-16')
    clip_pretrained  : open_clip pretrained weights  (default 'openai')
    clip_dim         : CLIP embedding dimension (512 for ViT-B/16)
    train_batch_size : images per batch
    lr               : AdamW learning rate
    num_epochs       : training epochs
    amp              : use fp16 mixed precision
    num_workers      : DataLoader workers (defaults to min(cpu_count, 8))
    results_folder   : where to save aligner checkpoints
    """

    def __init__(
        self,
        encoder,
        folder,
        *,
        source_transform = None,
        clip_model_name  = 'ViT-B-16',
        clip_pretrained  = 'openai',
        clip_dim         = 512,
        train_batch_size = 256,
        lr               = 1e-3,
        num_epochs       = 10,
        amp              = False,
        num_workers      = None,
        results_folder   = './results',
    ):
        import open_clip
        clip_model, _, clip_preprocess_val = open_clip.create_model_and_transforms(
            clip_model_name, pretrained = clip_pretrained
        )
        self.clip_model = clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        clip_transform   = clip_preprocess_val
        source_transform = default(source_transform, clip_preprocess_val)
        source_dim       = infer_encoder_dim(encoder)

        ds = PairedTransformDataset(folder, source_transform, clip_transform)
        dl = DataLoader(
            ds,
            batch_size  = train_batch_size,
            shuffle     = True,
            num_workers = default(num_workers, min(cpu_count(), 8)),
            pin_memory  = True,
            drop_last   = True,
        )

        self.aligner        = Text2ConceptAligner(source_dim, clip_dim)
        self.encoder        = encoder
        self.opt            = AdamW(self.aligner.parameters(), lr = lr)
        self.num_epochs     = num_epochs
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self._clip_model_name = clip_model_name
        self._clip_pretrained  = clip_pretrained

        self.accelerator = Accelerator(mixed_precision = 'fp16' if amp else 'no')
        self.encoder, self.aligner, self.opt, self.dl = \
            self.accelerator.prepare(self.encoder, self.aligner, self.opt, dl)
        self.clip_model = self.clip_model.to(self.accelerator.device)

    @property
    def device(self):
        return self.accelerator.device

    def train(self):
        self.encoder.eval()
        self.clip_model.eval()

        for epoch in range(self.num_epochs):
            total_loss  = 0.
            num_batches = 0

            with tqdm(
                self.dl,
                desc    = f'Epoch {epoch + 1}/{self.num_epochs}',
                disable = not self.accelerator.is_main_process,
            ) as pbar:
                for source_imgs, clip_imgs in pbar:
                    with torch.no_grad():
                        source_feats = flatten_features(self.encoder(source_imgs))
                        clip_feats   = l2norm(self.clip_model.encode_image(clip_imgs))

                    aligned = self.aligner(source_feats)
                    # cosine loss: 0 = perfectly aligned, 2 = anti-aligned
                    loss = (1 - (aligned * clip_feats).sum(dim = -1)).mean()

                    self.accelerator.backward(loss)
                    self.opt.step()
                    self.opt.zero_grad()

                    total_loss  += loss.item()
                    num_batches += 1
                    pbar.set_postfix(loss = f'{loss.item():.4f}')

            avg = total_loss / num_batches
            self.accelerator.print(
                f'Epoch {epoch + 1}/{self.num_epochs} | avg cosine loss: {avg:.4f}'
            )

        self.save('final')

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        aligner = self.accelerator.unwrap_model(self.aligner)
        data = {
            'aligner'    : self.accelerator.get_state_dict(self.aligner),
            'source_dim' : aligner.source_dim,
            'clip_dim'   : aligner.clip_dim,
        }
        torch.save(data, str(self.results_folder / f'aligner-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(
            str(self.results_folder / f'aligner-{milestone}.pt'),
            map_location = self.device,
            weights_only = True,
        )
        self.accelerator.unwrap_model(self.aligner).load_state_dict(data['aligner'])

    def get_text2concept(self, templates = None):
        """Convenience: return a ready-to-use Text2Concept from the trained aligner."""
        return Text2Concept(
            self.accelerator.unwrap_model(self.encoder),
            self.accelerator.unwrap_model(self.aligner),
            clip_model_name = self._clip_model_name,
            clip_pretrained  = self._clip_pretrained,
            templates        = templates,
        )
