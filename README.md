# TPP

Code and data for "Text-promptable Propagation for Referring Medical Image Sequence Segmentation". 

## Datasets
We curate a large and comprehensive benchmark for referring medical image sequence segmentation. The benchmark is sourced from 18 datasets consist of medical image sequences, including 20 anatomical structures across 4 different imaging modalities. Processed data with text prompts is coming soon.

**Note:** if you plan to use these datasets, be sure to follow the citation guidelines provided by the original authors.
| Dataset                                        | Class                                                                               | Modality   | Link                                                                                   | Processed data |
|------------------------------------------------|-------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------|:----------------:|
| 2018 Atria Segmentation Data                   | Left atria                                                                          | MRI        | [link](https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/)            |    upcoming    |
| RVSC                                           | Right ventricle                                                                     | MRI        | [link](https://rvsc.projets.litislab.fr/)                                              |    upcoming    |
| ACDC                                           | Left ventricle,  Myocardium,  Right ventricle                                       | MRI        | [link](https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html)                     |    upcoming    |
| CAMUS                                          | Left ventricle,  Myocardium,  Left atria                                            | Ultrasound | [link](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html)                |    upcoming    |
| NSCLC                                          | Lung                                                                                | CT         | [link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=68551327)    |    upcoming    |
| Spleen Segmentation Dataset                    | Spleen                                                                              | CT         | [link](https://ieeexplore.ieee.org/document/9112221)                                   |    upcoming    |
| Pancreas-CT                                    | Pancreas                                                                            | CT         | [link](https://www.cancerimagingarchive.net/collection/pancreas-ct/)                   |    upcoming    |
| BTCV                                           | Aorta,  Gallbladder,  Kidney (L),  Kidney (R),  Liver,  Pancreas,  Spleen,  Stomach | CT         | [link](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)                       |    upcoming    |
| Micro-Ultrasound Prostate Segmentation Dataset | Prostate                                                                            | Ultrasound | [link](https://github.com/mirthAI/MicroSegNet)                                         |    upcoming    |
| BraTS 2019                                     | Brain tumor                                                                         | MRI        | [link](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019) |    upcoming    |
| Breast_Cancer_DCE-MRI_Data                     | Breast mass                                                                         | MRI        | [link](https://zenodo.org/records/8068383)                                             |    upcoming    |
| RIDER                                          | Breast mass                                                                         | MRI        | [link](https://www.cancerimagingarchive.net/collection/rider-breast-mri/)              |    upcoming    |
| LiTS                                           | Liver tumor                                                                         | CT         | [link](https://competitions.codalab.org/competitions/17094)                            |    upcoming    |
| KiTS 2023                                      | Kidney tumor                                                                        | CT         | [link](https://kits-challenge.org/kits23/)                                             |    upcoming    |
| CVC-ClinicDB                                   | Polyp                                                                               | Endoscopy  |                                                                                        |    upcoming    |
| CVC-ColonDB                                    | Polyp                                                                               | Endoscopy  |                                                                                        |    upcoming    |
| ETIS                                           | Polyp                                                                               | Endoscopy  |                                                                                        |    upcoming    |
| ASU-Mayo                                       | Polyp                                                                               | Endoscopy  |                                                                                        |    upcoming    |


## Installation
We use `python=3.9` and recommend `torch 1.12.1`, `cuda 11.3`.

    conda create -n env_name python=3.9
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r requirements.txt

## Usage
Training

    # ResNet-50
    bash scripts/custom_r50.sh

    # Swin Tranformer-Large
    bash scripts/custom_swinl.sh

    # Video Swin Transformer-Tiny
    bash scripts/custom_vswint.sh

