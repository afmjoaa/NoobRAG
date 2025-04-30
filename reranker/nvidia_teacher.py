from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document
import json

query = "What is the GPU memory bandwidth of H100 SXM?"
passages = [
    "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to 10X higher performance for applications running terabytes of data.",
    "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets.",
    "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™.",
]

client = NVIDIARerank(
  model="nvidia/nv-rerankqa-mistral-4b-v3",
  api_key="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",
)

response = client.compress_documents(
  query=query,
  documents=[Document(page_content=passage) for passage in passages]
)
print(response)
print(response[0].metadata['relevance_score'])
print(f"Most relevant: {response[0].page_content}\nLeast relevant: {response[-1].page_content}")
