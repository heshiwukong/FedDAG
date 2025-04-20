# FedDAG: Dynamic Adjacency Graph for Personalized Federated Learning

## 1. Introduction
In real-world federated learning (FL), data heterogeneity across clients often leads to slower convergence, reduced accuracy, and diminished collaboration. FedDNG proposes a novel PFL approach based on dynamic neighbors graphs that:
1. Constructs/updates dynamic graphs to capture client collaboration relationships
2. Employs graph-based knowledge fusion for local model optimization
3. Demonstrates strong adaptability across 8 datasets and 8 heterogeneity scenarios

## 2. Dataset Processing
### 2.1 Supported Datasets
| Dataset | Type | Classes | Samples | Characteristics |
|---------|------|---------|---------|------------------|
| [PAMAP2](https://wjdcloud.blob.core.windows.net/dataset/cycfed/pamap.tar.gz) | Sensor | 18 | 2.84M | Human activity recognition |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | Image | 10 | 60K | Object recognition benchmark |
| [FEMNIST](https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz) | Handwriting | 62 | 814K | Natural writer heterogeneity |
| [PACS](https://wjdcloud.blob.core.windows.net/dataset/PACS.zip) | Image | 7 | 9,991 | Cross-domain style variations |
| [OrganMNISTA](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistA.tar.gz) | Medical | 11 | 58,850 | Axial CT view |
| [OrganMNISTC](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistC.tar.gz) | Medical | 11 | 23,660 | Coronal CT view |
| [OrganMNISTS](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnist.tar.gz) | Medical | 11 | 25,221 | Sagittal CT view |
| [COVID-19](https://wjdcloud.blob.core.windows.net/dataset/cycfed/covid19.tar.gz) | Medical | 4 | 9,208 | Pneumonia classification |

### 2.2 Data Partition Strategies
We implement heterogeneous data scenarios covering three main categories of data shifts:

#### Feature Distribution Skew
- **Noise-based Feature Imbalance**  
  Equal random partition with different Gaussian noise levels injected per client to create artificial feature shifts.

- **Real-world Feature Imbalance (FEMNIST)**  
  Natural writer-based partitioning where each client exclusively contains handwritten characters from specific users.

- **Domain Feature Imbalance (PACS)**  
  Domain-specific allocation where each client corresponds to one data domain (Photos/Art/Cartoon/Sketch).

#### Label Distribution Skew
- **Dirichlet Label Distribution**  
  Non-IID label allocation using Dirichlet distribution to control label concentration across clients.

- **Fixed-class Label Allocation**  
  Each client receives data from exactly K classes, with class-balanced sample distribution within selected classes.

#### Quantity Skew
- **Dirichlet Quantity Allocation**  
  IID data distribution with client dataset sizes determined by Dirichlet distribution.

- **Long-tail Quantity Allocation**  
  All-class preservation with dominant-class emphasis - each client maintains all classes but has one dominant class containing majority samples.

#### Mixed Shifts
- **Feature+Label Hybrid**  
  Combines noise-based feature distortion with Dirichlet label distribution.

- **Feature+Quantity Hybrid**  
  Integrates feature noise injection with long-tail quantity allocation.

## 3. Installation & Usage
### Quick Start
```bash
# Clone repository
git clone https://github.com/heshiwukong/FedDNG.git
cd FedDNG

# Install dependencies
pip install -r requirements.txt

# Run 
bash run.sh
```
### Key Parameters Configuration
#### Core Training Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--alg` | str | "FedDNG" | Algorithm to learning|
| `--dataset` | str | "pamap" | Dataset selection (pamap/cifar10/femnist etc) |
| `--partition` | str | "noniid-label-dir" | Data partition strategy |
| `--n_parties` | int | 20 | Number of clients in federation |
| `--global_iters` | int | 50 | Communication rounds |
| `--local_iters` | int | 10 | Local training iterations per round |
| `--lr` | float | 0.01 | Learning rate |
| `--device` | str | "cuda:1" | Computation device |

#### Data Partitioning Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--beta` | float | 0.1 | Dirichlet distribution concentration parameter |
| `--main_prop` | float | 0.8 | Dominant class proportion in long-tail distribution |
| `--noise_level` | float | 1.0 | Gaussian noise intensity (0-1 scale) |

## 4. References
This project builds upon previous work:
- https://github.com/TsingZ0/PFL-Non-IID
- https://github.com/microsoft/PersonalizedFL
- https://github.com/Xtra-Computing/NIID-Bench
- https://github.com/MediaBrain-SJTU/pFedGraph
