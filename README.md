<img src="./images/diagram.png" width="600px"></img>

## Text2Concept — in PyTorch

[![PyPI version](https://badge.fury.io/py/text2concept-pytorch.svg)](https://badge.fury.io/py/text2concept-pytorch)

Implementation of [Text2Concept: Concept Activation Vectors Directly from Text](https://openaccess.thecvf.com/content/CVPR2023W/XAI4CV/papers/Moayeri_Text2Concept_Concept_Activation_Vectors_Directly_From_Text_CVPRW_2023_paper.pdf) (CVPR Workshop 2023) in PyTorch.

Obtain Concept Activation Vectors for **any** vision encoder directly from text — no labelled concept data required.

## Real Training Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedFarag21/text2concept-pytorch/blob/main/real_test.ipynb)

Train on CIFAR-10 for 30 epochs — see `real_test.ipynb` for the full example including zero-shot classification, concept similarity, and distribution shift diagnosis.

## Install

```bash
$ pip install text2concept-pytorch
```

## How It Works

Text2Concept trains a single linear layer to align your encoder's feature space with CLIP's shared vision-language space. Once aligned, any text description can be compared directly to image features via cosine similarity.

```
text concept ──► CLIP text encoder ──► concept vector ─┐
                                                        ▼ cosine similarity
image ──► your encoder ──► linear aligner ──► aligned features
```

## Usage

### Step 1 — Train the aligner

Point it at a folder of images (ImageNet training set recommended).
Pass any `nn.Module` as the encoder — ResNet, ViT, DINO, anything.

```python
import torch
import torchvision.models as models
from text2concept_pytorch import Trainer

# any frozen vision encoder
encoder = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
encoder.fc = torch.nn.Identity()   # remove classification head
encoder.eval()

trainer = Trainer(
    encoder,
    'path/to/imagenet/train/',
    clip_model_name  = 'ViT-B-16',
    clip_pretrained  = 'openai',
    train_batch_size = 256,
    lr               = 1e-3,
    num_epochs       = 10,
    amp              = True,
    results_folder   = './results',
)

trainer.train()
# saves results/aligner-final.pt
```

### Step 2 — Query concepts

```python
from text2concept_pytorch import Text2Concept, Text2ConceptAligner

aligner = Text2ConceptAligner.from_checkpoint('results/aligner-final.pt')
t2c     = Text2Concept(encoder, aligner)

# cosine similarity to arbitrary text concepts  →  (batch, num_concepts)
sims = t2c.concept_similarity(images, ['red food', 'in a tree', 'indoors'])

# zero-shot classification
preds = t2c.zero_shot_classify(images, ['cat', 'dog', 'bird', 'car'])

# Concept Activation Vector in source encoder space (for TCAV-style analysis)
cav = t2c.get_cav('stripes')                     # (1, source_dim)
cav = t2c.get_cav('in a tree', class_names = imagenet_classes)  # class-refined
```

### Concept Bottleneck Model (no concept supervision)

```python
import torch.nn as nn

# 1. define your concept vocabulary
concepts = ['red', 'striped', 'furry', 'indoors', 'small', ...]

# 2. compute concept similarities as features
concept_features = t2c.concept_similarity(all_images, concepts)   # (N, num_concepts)

# 3. fit a linear head on top
cbm_head = nn.Linear(len(concepts), num_classes)
# train cbm_head on (concept_features, labels) with cross-entropy
```

### Distribution Shift Diagnosis

```python
train_sims = t2c.concept_similarity(train_images, ['indoors'])
test_sims  = t2c.concept_similarity(test_images,  ['indoors'])

# large KS statistic → 'indoors' is a key distribution shift factor
from scipy.stats import ks_2samp
stat, p = ks_2samp(train_sims[:, 0].cpu(), test_sims[:, 0].cpu())
```

## Multi-GPU

```bash
$ accelerate config
$ accelerate launch train.py
```

## Citations

```bibtex
@inproceedings{moayeri2023text2concept,
    title     = {Text2Concept: Concept Activation Vectors Directly from Text},
    author    = {Moayeri, Mazda and Rezaei, Keivan and Sanjabi, Maziar and Feizi, Soheil},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
    year      = {2023}
}
```
