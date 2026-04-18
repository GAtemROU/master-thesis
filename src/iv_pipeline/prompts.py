TASK_PROMPT = """You are solving a math problem.
Provide a brief solution and put the final answer on the last line as:
FINAL: <answer>
Problem: {question}
"""

CONSTRAINT_PROMPT = """List domain- or task-related constraints that can be used
to verify a solution. Return each constraint on its own line and start with "- ".
Problem: {question}
"""

VERIFY_PROMPT = """Verify the proposed solution using the constraints.
Return:
VERDICT: PASS or VERDICT: FAIL
REASON: <short reason>

Problem: {question}
Proposed Solution:
{solution}

Constraints:
{constraints}
"""
