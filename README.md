# LLM Integration & Prompt Engineering Suite

A robust Python implementation demonstrating enterprise-ready OpenAI LLM API integration featuring structured prompting, streaming, retry logic, output validation, and cost tracking.

## Features
- **Secure Key Management**: Environment-based API keys (`python-dotenv`).
- **Structured Prompting**: System + User role segregation with few-shot examples.
- **Streaming Support**: Real-time token streaming output.
- **Error Handling**: Retries with exponential backoff (`tenacity`) for rate limits and connection errors.
- **Strict Output Validation**: Pydantic schema parsing & JSON sanitization.
- **Cost Awareness**: Accurate token usage logging and per-request cost estimations.

## Setup Instructions

1. Clone repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>