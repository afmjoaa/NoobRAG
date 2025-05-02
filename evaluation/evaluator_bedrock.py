import boto3
import os
import re
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# --- Bedrock Configuration ---
AWS_REGION = os.getenv("AWS_REGION")
if not AWS_REGION:
  raise ValueError("AWS_REGION not found in environment variables. Please set it.")

BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Claude 3.5 Sonnet
# BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0" # Or Claude 3 Sonnet if needed

# Initialize Bedrock Runtime client
# It's generally recommended to create clients outside the threaded function
# if they are thread-safe or manage connections appropriately.
# Boto3 clients are generally thread-safe.
bedrock_runtime_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION
)

# --- Rate Limiting Considerations ---
# Bedrock has account/region level quotas. Boto3 SDK handles basic retries.
# For more robust handling, consider libraries like 'tenacity'.
# This script relies on SDK retries and basic error handling.
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds

# --- Evaluation Prompts (same as before) ---
RELEVANCE_PROMPT_TEMPLATE = """
You are an evaluator for the LiveRAG challenge. Your task is to assess the relevance of a generated answer to a given question. Use ONLY the following scale and definitions:
2: The response correctly answers the user question and contains no irrelevant content.
1: The response provides a useful answer to the user question, but may contain irrelevant content that does not harm the usefulness of the answer.
0: No answer is provided in the response (e.g., "I don’t know").
-1: The response does not answer the question whatsoever.

Question: {question}
Answer: {answer}

Based ONLY on the scale above, what is the relevance score for this answer? Respond with ONLY the integer score (-1, 0, 1, or 2) and nothing else.
"""

FAITHFULNESS_PROMPT_TEMPLATE = """
You are an evaluator for the LiveRAG challenge. Your task is to assess the faithfulness of a generated answer based ONLY on the provided context (retrieved passages). Faithfulness means whether the information in the answer is supported by the context. Use ONLY the following scale and definitions:
1: Full support, all answer parts are grounded in the context.
0: Partial support, not all answer parts are grounded in the context.
-1: No support, all answer parts are not grounded in the context.

Context:
---
{context}
---
Question: {question}
Answer: {answer}

Based ONLY on the scale above and the provided context, what is the faithfulness score for this answer? Respond with ONLY the integer score (-1, 0, or 1) and nothing else.
"""

# --- Evaluation Function ---


def get_llm_evaluation_score(prompt: str, model_id: str = BEDROCK_MODEL_ID) -> int | None:
  """Sends a prompt to the Bedrock model and attempts to parse an integer score."""
  retries = 0
  backoff = INITIAL_BACKOFF
  while retries < MAX_RETRIES:
    try:
      # Construct the payload for Claude 3 models
      # Note: Claude 3.5 uses the same messages API structure
      messages = [{"role": "user", "content": prompt}]
      body = json.dumps({
          "anthropic_version": "bedrock-2023-05-31",  # Required for Claude 3/3.5
          "max_tokens": 10,  # Expecting only a short score
          "messages": messages,
          "temperature": 0.0,  # Deterministic for evaluation
      })

      response = bedrock_runtime_client.invoke_model(
          body=body,
          modelId=model_id,
          accept='application/json',
          contentType='application/json'
      )

      response_body = json.loads(response.get('body').read())

      # Extract content for Claude 3/3.5 response structure
      if response_body.get("content") and isinstance(response_body["content"], list) and len(response_body["content"]) > 0:
        response_text = response_body["content"][0].get("text", "")
      else:
        print(f"Warning: Unexpected response format from Bedrock: {response_body}")
        response_text = ""

      # Attempt to extract the first integer found in the response text
      match = re.search(r'-?\d+', response_text.strip())
      if match:
        return int(match.group(0))
      else:
        print(f"Warning: Could not parse integer score from Bedrock response: {response_text}")
        return None  # Or retry if it seems like a transient parsing issue

    except ClientError as e:
      error_code = e.response.get("Error", {}).get("Code")
      if error_code == 'ThrottlingException' and retries < MAX_RETRIES - 1:
        print(f"Bedrock throttling detected. Retrying in {backoff} seconds...")
        time.sleep(backoff)
        backoff *= 2  # Exponential backoff
        retries += 1
      else:
        print(f"Error during Bedrock API call: {e}")
        return None  # Non-retryable error or max retries reached
    except Exception as e:
      print(f"Unexpected error during Bedrock call: {e}")
      return None  # Unexpected error

  print(f"Max retries reached for Bedrock call.")
  return None


def evaluate_answer(question: str, answer: str, context: str) -> dict:
  """
  Evaluates the answer based on relevance and faithfulness using Bedrock Claude 3.5 Sonnet sequentially.

  Args:
      question: The user's question.
      answer: The generated answer.
      context: The retrieved context passages used to generate the answer.

  Returns:
      A dictionary containing 'relevance_score' and 'faithfulness_score'.
      Scores can be None if evaluation fails.
  """
  relevance_prompt = RELEVANCE_PROMPT_TEMPLATE.format(question=question, answer=answer)
  faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
      context=context, question=question, answer=answer)

  relevance_score = None
  faithfulness_score = None

  # Evaluate sequentially
  print(f"Evaluating Relevance (using Bedrock {BEDROCK_MODEL_ID})...")
  try:
    relevance_score = get_llm_evaluation_score(relevance_prompt)
  except Exception as e:
    print(f"Error retrieving relevance score: {e}")

  print(f"Evaluating Faithfulness (using Bedrock {BEDROCK_MODEL_ID})...")
  try:
    faithfulness_score = get_llm_evaluation_score(faithfulness_prompt)
  except Exception as e:
    print(f"Error retrieving faithfulness score: {e}")

  return {
      "relevance_score": relevance_score,
      "faithfulness_score": faithfulness_score
  }


# --- Example Usage (same examples as before) ---
if __name__ == "__main__":
  # Ensure AWS credentials and region are configured (e.g., via env vars, ~/.aws/credentials, IAM role)
  print(f"Using AWS Region: {AWS_REGION}")
  print(f"Using Bedrock Model ID: {BEDROCK_MODEL_ID}")

  # Example 1
  example_question = "What is the capital of France?"
  example_answer = "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture."
  example_context = """
    Passage 1: Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region, also known as the région parisienne.
    Passage 2: France is a country located in Western Europe. It is known for its wines and sophisticated cuisine. Landmarks include the Eiffel Tower and the Louvre Museum.
    """

  print(f"\n--- Example 1 ---")
  print(f"Question: {example_question}")
  print(f"Answer: {example_answer}")
  print(f"Context:\n{example_context}\n")

  start_time = time.time()
  scores = evaluate_answer(example_question, example_answer, example_context)
  end_time = time.time()

  print("\n--- Evaluation Results 1 ---")
  print(f"Relevance Score: {scores['relevance_score']}")
  print(f"Faithfulness Score: {scores['faithfulness_score']}")
  print(f"Evaluation took {end_time - start_time:.2f} seconds.")

  # Example 2: Irrelevant but somewhat useful
  example_question_2 = "How do I bake a cake?"
  example_answer_2 = "Baking involves mixing ingredients like flour and sugar. Paris is the capital of France."
  example_context_2 = "Cake baking requires flour, sugar, eggs, and butter. Mix dry ingredients first, then wet."

  print("\n--- Example 2 ---")
  print(f"Question: {example_question_2}")
  print(f"Answer: {example_answer_2}")
  print(f"Context:\n{example_context_2}\n")
  start_time_2 = time.time()
  scores_2 = evaluate_answer(example_question_2, example_answer_2, example_context_2)
  end_time_2 = time.time()
  print("\n--- Evaluation Results 2 ---")
  print(f"Relevance Score: {scores_2['relevance_score']}")  # Expecting 1
  print(f"Faithfulness Score: {scores_2['faithfulness_score']}")  # Expecting 0
  print(f"Evaluation took {end_time_2 - start_time_2:.2f} seconds.")

  # Example 3: Not faithful
  example_question_3 = "What color is the sky?"
  example_answer_3 = "The sky is green according to this document."
  example_context_3 = "The sky appears blue due to Rayleigh scattering."

  print("\n--- Example 3 ---")
  print(f"Question: {example_question_3}")
  print(f"Answer: {example_answer_3}")
  print(f"Context:\n{example_context_3}\n")
  start_time_3 = time.time()
  scores_3 = evaluate_answer(example_question_3, example_answer_3, example_context_3)
  end_time_3 = time.time()
  print("\n--- Evaluation Results 3 ---")
  print(f"Relevance Score: {scores_3['relevance_score']}")  # Expecting 2 or 1
  print(f"Faithfulness Score: {scores_3['faithfulness_score']}")  # Expecting -1
  print(f"Evaluation took {end_time_3 - start_time_3:.2f} seconds.")

  # Example to test potential throttling over multiple calls
  print("\n--- Throttling Test (may take time if limits hit) ---")
  for i in range(6):  # Making 6 * 2 = 12 Bedrock calls
    print(f"Evaluation call {i+1}")
    start_time_t = time.time()
    scores_test = evaluate_answer(example_question, example_answer, example_context)
    end_time_t = time.time()
    print(f"Call {i+1} results: {scores_test}, Time: {end_time_t - start_time_t:.2f}s")
    # No explicit sleep, relying on potential backoff in get_llm_evaluation_score
