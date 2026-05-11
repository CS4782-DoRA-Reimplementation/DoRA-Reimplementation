# Data

This project uses the following benchmark datasets through the Hugging Face
`datasets` library:

- BoolQ: `boolq`
- PIQA: `piqa`
- Social IQa: `social_i_qa`
- HellaSwag: `hellaswag`
- ARC-Challenge: `ai2_arc`, config `ARC-Challenge`
- ARC-Easy: `ai2_arc`, config `ARC-Easy`
- WinoGrande: `winogrande`, config `winogrande_xl`
- OpenBookQA: `openbookqa`, config `main`

The datasets are downloaded automatically by `code/train.py` and
`code/evaluate.py` when the scripts are run. Raw dataset files are not committed
to this repository.

To obtain the data, install the dependencies and run one of the training or
evaluation commands in the root `README.md`.
