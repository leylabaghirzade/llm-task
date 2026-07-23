import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, APIError, RateLimitError, APIConnectionError
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Validate API Key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is missing!")

# Client initialization pointing directly to Groq's OpenAI-compatible endpoint
client = OpenAI(
    api_key=API_KEY, 
    base_url="https://api.groq.com/openai/v1"
)

# -------------------------------------------------------------------
# Checkpoint 5: Pydantic Schema for Structured Output Validation
# -------------------------------------------------------------------
class SummaryResponse(BaseModel):
    summary: str
    key_points: list[str]
    sentiment: str

# -------------------------------------------------------------------
# Checkpoint 4 & Quality Check: Error Handling + Exponential Backoff Retry
# -------------------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError))
)
def call_llm_with_retry(
    messages: list, 
    stream: bool = False, 
    model: str = "llama-3.1-8b-instant",
    json_mode: bool = False
):
    """API Call wrapped with automatic retry for transient errors."""
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "stream": stream
        }
        # Force JSON response from Llama 3.1 when structured data is requested
        if json_mode and not stream:
            kwargs["response_format"] = {"type": "json_object"}

        return client.chat.completions.create(**kwargs)
    except OpenAIError as e:
        logger.error(f"OpenAI API Call Failed: {e}")
        raise e

# -------------------------------------------------------------------
# Checkpoint 1 & 2: API Integration & Structured Prompting
# -------------------------------------------------------------------
def generate_analysis(user_text: str) -> Optional[SummaryResponse]:
    system_prompt = (
        "You are an expert content analyzer. Analyze the user text and return a valid JSON object matching this schema:\n"
        "{\n"
        '  "summary": "brief summary",\n'
        '  "key_points": ["point 1", "point 2"],\n'
        '  "sentiment": "positive | neutral | negative"\n'
        "}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze the following text:\n{user_text}"}
    ]

    logger.info("Sending request to LLM...")

    # Checkpoint 1 & 4 Call
    response = call_llm_with_retry(messages, stream=False, json_mode=True)

    # -------------------------------------------------------------------
    # Checkpoint 6: Token Usage & Cost Estimation Awareness
    # -------------------------------------------------------------------
    if hasattr(response, 'usage') and response.usage:
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        # Groq Llama 3.1 8B Instant Pricing ($0.05/1M input, $0.08/1M output)
        est_cost = (prompt_tokens * 0.00000005) + (completion_tokens * 0.00000008)
        
        logger.info(f"[Token Usage] Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
        logger.info(f"[Estimated Cost] ${est_cost:.8f}")

    raw_content = response.choices[0].message.content

    # -------------------------------------------------------------------
    # Checkpoint 5 & Quality Check: Output Parsing & Validation
    # -------------------------------------------------------------------
    try:
        # Sanitize potential leftover markdown formatting
        clean_content = raw_content.strip()
        if clean_content.startswith("```"):
            lines = clean_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_content = "\n".join(lines).strip()

        parsed_json = json.loads(clean_content)
        validated_data = SummaryResponse(**parsed_json)
        return validated_data

    except (json.JSONDecodeError, ValidationError) as err:
        logger.error(f"[Validation Error] Failed to parse or validate JSON output: {err}")
        logger.debug(f"Raw Output was: {raw_content}")
        return None

# -------------------------------------------------------------------
# Checkpoint 3: Streaming Response Management
# -------------------------------------------------------------------
def stream_llm_response(prompt: str):
    logger.info("\n--- Streaming Response Start ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    response_stream = call_llm_with_retry(messages, stream=True)
    
    for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print("\n--- Streaming Response End ---\n")

# -------------------------------------------------------------------
# Execution / Test Demo
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The space mission successfully launched at 08:00 UTC. All systems are operating nominal, "
        "and telemetry confirms payload orbit injection. The team reported slight delays during "
        "pre-flight checks, but no critical issues occurred."
    )

    print("\n=== Test 1: Structured Execution with Validation & Cost Tracking ===")
    result = generate_analysis(sample_text)
    if result:
        print("Validated Output Object:", result.model_dump())

    print("\n=== Test 2: Streaming Response Execution ===")
    stream_llm_response("Briefly explain how neural networks learn.")