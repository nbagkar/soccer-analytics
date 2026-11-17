"""Optional LLM layer for the assistant: a narrator over answers the app has already computed.

Off unless SOCCER_LLM_PROVIDER, SOCCER_LLM_MODEL and SOCCER_LLM_API_KEY are set. The rule-based
assistant stays the source of every number; the LLM only rewrites a computed answer, and a
reply mentioning a figure the computed answer doesn't contain is discarded.
"""
