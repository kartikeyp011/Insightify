"""
Scratchpad script for locally testing generation of logical questions.

This is fundamentally a throwaway/diagnostic module to verify prompt handling
when extracting outputs from the QA engine logic without running the full API server.

Dependencies:
    - backend.utils.qa_engine: Relies on core backend implementations.
"""
from backend.utils.qa_engine import generate_logic_questions

# Call the generator logic locally to print raw output directly for evaluation
print(generate_logic_questions())