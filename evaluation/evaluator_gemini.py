import google.genai as genai  # Updated import
import os
import re
import time
import threading  # Added for locking
import itertools  # Added for cycling keys
from collections import deque
from concurrent.futures import ThreadPoolExecutor  # Added for parallel execution
from dotenv import load_dotenv
from google.genai import types as genai_types  # For potential error handling or types
from google.api_core import exceptions as api_exceptions  # More specific exceptions

# Load environment variables
load_dotenv()

# Configure the Gemini client - Get multiple keys
api_keys_str = os.getenv("GOOGLE_API_KEYS")  # Expecting comma-separated keys
if not api_keys_str:
  raise ValueError(
      "GOOGLE_API_KEYS not found in environment variables. Please set it in a .env file or environment as a comma-separated string."
  )
API_KEYS = [key.strip() for key in api_keys_str.split(',') if key.strip()]
if not API_KEYS:
  raise ValueError("No valid API keys found in GOOGLE_API_KEYS environment variable.")

# Use itertools.cycle to rotate through keys for distribution
key_cycler = itertools.cycle(API_KEYS)

# --- Rate Limiting Globals (Per Key) ---
REQUEST_LIMIT = 10
TIME_WINDOW_SECONDS = 60
# Dictionary to store timestamps deque for each API key
key_request_timestamps = {key: deque() for key in API_KEYS}
rate_limit_lock = threading.Lock()  # Lock for thread-safe access to timestamps

# --- Evaluation Prompts based on LiveRAG Guidelines ---
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


def _apply_rate_limit(api_key: str):
  """Checks and enforces the rate limit for a specific API key."""
  global key_request_timestamps, rate_limit_lock
  current_time = time.monotonic()

  with rate_limit_lock:  # Ensure thread-safe access
    timestamps = key_request_timestamps[api_key]

    # Remove timestamps older than the time window
    while timestamps and current_time - timestamps[0] > TIME_WINDOW_SECONDS:
      timestamps.popleft()

    # If limit reached, calculate wait time and sleep
    if len(timestamps) >= REQUEST_LIMIT:
      time_since_oldest_request = current_time - timestamps[0]
      wait_time = TIME_WINDOW_SECONDS - time_since_oldest_request + 0.1  # Add small buffer
      print(
          f"Rate limit reached for key ending in ...{api_key[-4:]}. Waiting for {wait_time:.2f} seconds...")
      # Release lock before sleeping to allow other threads with different keys to proceed
      rate_limit_lock.release()
      try:
        time.sleep(wait_time)
      finally:
        # Reacquire lock after sleeping before modifying the deque again
        rate_limit_lock.acquire()
      # Re-evaluate current time after sleeping
      current_time = time.monotonic()
      # Re-check timestamps after waiting, as other threads might have added some
      timestamps = key_request_timestamps[api_key]  # Re-fetch in case it changed
      while timestamps and current_time - timestamps[0] > TIME_WINDOW_SECONDS:
        timestamps.popleft()

    # Record the timestamp of the upcoming request
    timestamps.append(current_time)


def get_llm_evaluation_score(prompt: str, api_key: str, model_name: str = "gemini-1.5-flash") -> int | None:
  """Sends a prompt to the Gemini model using a specific API key and attempts to parse an integer score, respecting rate limits."""
  try:
    _apply_rate_limit(api_key)  # Enforce rate limit for the specific key

    # Create a client instance for this specific call using the assigned API key
    client = genai.Client(api_key=api_key)

    # Use the client to generate content
    response = client.models.generate_content(model=model_name, contents=prompt)

    # Attempt to extract the first integer found in the response text
    # Accessing response text might differ slightly, adjust if needed based on actual response object
    # Assuming response.text still works based on common patterns
    match = re.search(r'-?\d+', response.text.strip())
    if match:
      return int(match.group(0))
    else:
      print(
          f"Warning: Could not parse integer score from response (key ...{api_key[-4:]}): {response.text}")
      return None
  # Catch more specific API errors if possible
  except (api_exceptions.GoogleAPIError, genai_types.BlockedPromptError, genai_types.StopCandidateError) as e:
    print(f"API Error during Gemini API call (key ...{api_key[-4:]}): {type(e).__name__} - {e}")
    return None
  except Exception as e:
    print(f"Generic Error during Gemini API call (key ...{api_key[-4:]}): {type(e).__name__} - {e}")
    return None


def evaluate_answer(question: str, answer: str, context: str) -> dict:
  """
  Evaluates the answer based on relevance and faithfulness using Gemini in parallel,
  distributing requests across available API keys.

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

  # Use ThreadPoolExecutor for parallel execution
  # Max workers set to number of keys to potentially utilize all keys simultaneously
  with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
    # Assign API keys using the cycler for better distribution across calls
    relevance_key = next(key_cycler)
    faithfulness_key = next(key_cycler)

    print(f"Evaluating Relevance (using key ...{relevance_key[-4:]})...")
    future_relevance = executor.submit(get_llm_evaluation_score, relevance_prompt, relevance_key)

    print(f"Evaluating Faithfulness (using key ...{faithfulness_key[-4:]})...")
    future_faithfulness = executor.submit(
        get_llm_evaluation_score, faithfulness_prompt, faithfulness_key)

    try:
      relevance_score = future_relevance.result()
    except Exception as e:
      print(f"Error retrieving relevance score: {e}")

    try:
      faithfulness_score = future_faithfulness.result()
    except Exception as e:
      print(f"Error retrieving faithfulness score: {e}")

  return {
      "relevance_score": relevance_score,
      "faithfulness_score": faithfulness_score
  }


# --- Example Usage ---
if __name__ == "__main__":
    # Define the output log file
  log_filename = "evaluation_log.txt"

  # Function to write evaluation details to the log file
  def log_evaluation(filename, q, a, c, scores, call_num=None):
    with open(filename, 'a', encoding='utf-8') as f:
      if call_num:
        f.write(f"--- Evaluation Call {call_num} ---\n")
      else:
        f.write("--- Evaluation Run ---\n")
      f.write(f"Question: {q}\n")
      f.write(f"Answer: {a}\n")
      f.write(f"Context:\n{c}\n")
      f.write(f"Relevance Score: {scores['relevance_score']}\n")
      f.write(f"Faithfulness Score: {scores['faithfulness_score']}\n")
      f.write("-" * 20 + "\n\n")

  # Example 1
  example_question = "What is the capital of France?"
  example_answer = "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture."
  example_context = """
    Passage 1: Paris is the capital and most populous city of France. Situated on the Seine River, in the north of the country, it is in the centre of the Île-de-France region, also known as the région parisienne.
    Passage 2: France is a country located in Western Europe. It is known for its wines and sophisticated cuisine. Landmarks include the Eiffel Tower and the Louvre Museum.
    """
  print(f"Using {len(API_KEYS)} API keys for evaluation.")
  print(f"Logging results to {log_filename}")

  start_time = time.time()
  scores = evaluate_answer(example_question, example_answer, example_context)
  end_time = time.time()
  print(f"Example 1 evaluation took {end_time - start_time:.2f} seconds.")
  log_evaluation(log_filename, example_question, example_answer, example_context, scores)

  # Example 2: Irrelevant but somewhat useful
  example_question_2 = "How do I bake a cake?"
  example_answer_2 = "Baking involves mixing ingredients like flour and sugar. Paris is the capital of France."
  example_context_2 = "Cake baking requires flour, sugar, eggs, and butter. Mix dry ingredients first, then wet."
  start_time = time.time()
  scores_2 = evaluate_answer(example_question_2, example_answer_2, example_context_2)
  end_time = time.time()
  print(f"Example 2 evaluation took {end_time - start_time:.2f} seconds.")
  log_evaluation(log_filename, example_question_2, example_answer_2, example_context_2, scores_2)

  # Example 3: Not faithful
  example_question_3 = "What color is the sky?"
  example_answer_3 = "The sky is green according to this document."
  example_context_3 = "The sky appears blue due to Rayleigh scattering."
  start_time = time.time()
  scores_3 = evaluate_answer(example_question_3, example_answer_3, example_context_3)
  end_time = time.time()
  print(f"Example 3 evaluation took {end_time - start_time:.2f} seconds.")
  log_evaluation(log_filename, example_question_3, example_answer_3, example_context_3, scores_3)

  # Example to test rate limiting over multiple calls
  print("\n--- Rate Limit Test (Logging to file) ---")
  for i in range(6):
    print(f"Running evaluation call {i+1}...")
    start_time = time.time()
    scores_test = evaluate_answer(example_question, example_answer, example_context)
    end_time = time.time()
    print(f"Call {i+1} finished in {end_time - start_time:.2f}s. Logging results...")
    log_evaluation(log_filename, example_question, example_answer,
                   example_context, scores_test, call_num=i+1)

  print(f"\nFinished examples. All results logged to {log_filename}")
