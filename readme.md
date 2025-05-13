# RAG System Development for LiveRAG Challenge

## Team Members:
- Ramy Boulos
- Himanshu Manoj Kaloni
- A F M Mohimenul Joaa

## Development Checklist:


  - [x] Review and use [Indices Usage Examples](https://huggingface.co/spaces/LiveRAG/Challenge/blob/main/Operational_Instructions/Indices_Usage_Examples_for_LiveRAG.ipynb)
  - [x] Review and use [DM API Usage Example](https://huggingface.co/spaces/LiveRAG/Challenge/blob/main/Operational_Instructions/DM_API_usage_example.ipynb)
  - [x] Complete the basic flow of RAG

  - [ ] **Create a testing/benchmarking workflow** (Himanshu)

  - [ ] Figure out the best prompt strategy (We can only use 200 token max) (Ramy)
       - [ ] Experiment with different prompt formats
       - [ ] Test few-shot and zero-shot prompting
       - [ ] Evaluate prompt effectiveness on system performance
  - [ ] Add query refinement/expansion (Model selection) (Ramy)
  - [ ] Generating a plausible answer hypothesis (Model selection) (Ramy)
  - [ ] Add adaptive iteration control ***
  - [ ] Add Focus Mode Transformation Reranking (Joaa)
  - [ ] Add differentiable Gumbel Softmax reranking and diversity Reranking (Joaa)


## Resources:
- [LiveRAG Challenge Guidelines](https://liverag.tii.ae/challenge-guidelines.php)
- [HuggingFace Challenge Site](https://huggingface.co/spaces/LiveRAG/Challenge)
- [AI71 Platform](https://platform.ai71.ai/documentation)

## Action items:
- Meeting time need to be fixed


## Live Challenge Day logistics
- Date: May 12 – two time slots
- Test set: 500-1000 questions
- Time window: 3-4 hours
- Answer length: unlimited but only the first 300 words will be
evaluated
- Answers file: question, passages, final prompt, answer

- batch requesting falcon is possible see https://huggingface.co/spaces/LiveRAG/Challenge/blob/main/Operational_Instructions/Falcon_Ai71_Usage.ipynb

## Command to replicate the conda environment
`conda env create -f environment.yml`

## You can also use pip to install dependencies, after activating your conda environment
`pip install -r requirements.txt`


## **Batch with gpu cluster** -- This is used to generate answer for LiveRAG challenge (To get 500 or more questions answered)

## Batch with gpu (To get upto 100 questions answered)

## Single with cpu without query refinement (To get only one question answered)

## Basic RAG (To check if the basic rag system (Without query refinement, and reranking) works or not)


[//]: # (# Change the question path in batch_with_gpu_cluster.py and then run the cluster.sh)