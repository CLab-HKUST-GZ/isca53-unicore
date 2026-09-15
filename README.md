# Artifact Evaluation for ISCA53 UniCore

This repository contains the source code for reproducing the experiments in the paper ["UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference"](https://ieeexplore.ieee.org/abstract/document/11617931) at ISCA'26.

[`Hardware/UniCore`](./Hardware/UniCore) contains the hardware design of UniCore.

[`Software/Accuracy`](./Software/Accuracy) contains the UniCore framework with PyTorch.

[`Software/Simulator`](./Software/Simulator) contains the performance and energy evaluation of UniCore.

## Project Structure
```
isca53-unicore-ae/
├── README.md
├── Hardware/
│   ├── UniCore/
│   │   ├── README.md
│   │   ├── project/
│   │   ├── hw/
├── Software/
│   ├── Accuracy/
│   │   ├── README.md
│   │   ├── ae_scripts/
│   │   ├── quant_utils/
│   │   ├── unicore_kernel/
│   ├── Simulator/
│   │   ├── README.md
```