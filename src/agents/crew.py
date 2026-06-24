import os
import re
import json
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

_RETRYABLE_CODES = {503}
_QUOTA_CODES = {429}
_DEFAULT_RETRY_WAIT = 65.0


def _parse_retry_delay(error: Exception) -> float:
    # Read retryDelay from structured error body e.g. {'retryDelay': '57s'}
    try:
        body = getattr(error, 'response_json', None) or {}
        for detail in body.get('error', {}).get('details', []):
            raw = detail.get('retryDelay', '')
            if raw:
                return float(re.sub(r'[^\d.]', '', raw)) + 5.0
    except Exception:
        pass
    # Fallback: scan full string for patterns like "57s" or "57.8s"
    try:
        match = re.search(r'(\d+(?:\.\d+)?)s', str(error))
        if match:
            return float(match.group(1)) + 5.0
    except Exception:
        pass
    return _DEFAULT_RETRY_WAIT


def _generate_with_fallback(client: genai.Client, primary_model: str, **kwargs):
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]
    last_error = None

    for model_id in models_to_try:
        for attempt in range(1, 5):
            try:
                return client.models.generate_content(model=model_id, **kwargs)
            except (ServerError, ClientError) as e:
                if e.code in _QUOTA_CODES:
                    wait = _parse_retry_delay(e)
                    logger.warning(
                        "Model '%s' quota hit (429). Waiting %.0fs before trying next model...",
                        model_id, wait,
                    )
                    time.sleep(wait)
                    last_error = e
                    break
                elif e.code in _RETRYABLE_CODES:
                    last_error = e
                    time.sleep(2.0 * (2 ** (attempt - 1)))
                else:
                    raise

    raise RuntimeError(f"All models failed. Last error: {last_error}")


def _has_numeric_data(text: str) -> bool:
    patterns = [
        r'\[[\d,\s]+\]',
        r'\b\d{3,}\b.*\b\d{3,}\b',
        r'\b(error|failure|latency|rate|cpu|memory|lag)\b.*\d',
        r'\d+%',
        r'\d+ms',
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


class TicketAnalysis(BaseModel):
    urgency: str
    category: str
    summary: str


class AuditResult(BaseModel):
    approved: bool
    audit_notes: str


class TriageAgent:
    def __init__(self, client: genai.Client, model_id: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.model_id = model_id

    def analyze(self, description: str) -> dict:
        prompt = f"""
        You are an expert IT Helpdesk Triage Agent.
        Analyze the following support ticket and provide:
        1. urgency (Low, Medium, High - High strictly for production/server down)
        2. category (e.g., password_reset, software_install, server_down, hardware)
        3. summary (A clear, one-sentence summary of the issue)

        Ticket Description:
        {description}
        """
        response = _generate_with_fallback(
            self.client,
            self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TicketAnalysis,
                temperature=0.1,
            ),
        )
        return json.loads(response.text)


def _extract_parts(response) -> tuple[str, list[dict]]:
    final_text = ""
    execution_logs = []
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.executable_code:
                execution_logs.append({"type": "code", "content": part.executable_code.code})
            elif part.code_execution_result:
                execution_logs.append({"type": "output", "content": part.code_execution_result.output})
            elif part.text:
                final_text += part.text
    return final_text.strip(), execution_logs


class ResearcherAgent:
    def __init__(self, client: genai.Client, rag_mcp, model_id: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.rag_mcp = rag_mcp
        self.model_id = model_id

    def investigate(self, issue_summary: str, raw_description: str = "") -> dict:
        # Use raw_description for numeric detection and sandbox computation.
        # Use issue_summary (triage summary) only for RAG vector search.
        full_issue = raw_description if raw_description else issue_summary
        rag_context = self.rag_mcp.search_knowledge_base(issue_summary)
        requires_sandbox = _has_numeric_data(full_issue)

        if requires_sandbox:
            sandbox_instruction = """
        MANDATORY: This ticket contains numeric data (logs, arrays, rates, latencies).
        You MUST use the code_execution tool to compute exact values before responding.
        Do NOT estimate or calculate manually. Executing code is not optional."""
        else:
            sandbox_instruction = """
        Use the code_execution tool only if you encounter numeric data that requires computation."""

        system_prompt = f"""
        You are an expert Tier 3 IT Security & Research Engineer.
        Extract ONLY the exact technical steps or rules from the policy context that apply to the issue.
        {sandbox_instruction}
        Cite the source document for every rule: [SOURCE DOCUMENT: filename]
        Do NOT write an email. State facts only.

        User Issue:
        {full_issue}

        Enterprise Policy Context:
        {rag_context}
        """

        code_execution_config = types.GenerateContentConfig(
            temperature=0.1,
            tools=[{"code_execution": {}}],
        )

        response = _generate_with_fallback(
            self.client,
            self.model_id,
            contents=system_prompt,
            config=code_execution_config,
        )
        final_text, execution_logs = _extract_parts(response)

        if requires_sandbox and not execution_logs:
            history = [
                types.Content(role="user", parts=[types.Part(text=system_prompt)]),
                types.Content(role="model", parts=[types.Part(text=final_text or "(no response)")]),
                types.Content(role="user", parts=[types.Part(text=(
                    "PROTOCOL VIOLATION: You did not use the code_execution tool. "
                    "This ticket contains numeric data that MUST be computed via Python script. "
                    "Write and execute the script now. No exceptions."
                ))]),
            ]
            response2 = _generate_with_fallback(
                self.client,
                self.model_id,
                contents=history,
                config=code_execution_config,
            )
            final_text2, execution_logs2 = _extract_parts(response2)

            if not execution_logs2:
                raise RuntimeError(
                    "ResearcherAgent: Gemini refused to execute code after two turns."
                )

            final_text = final_text2
            execution_logs = execution_logs2

        return {
            "text": final_text,
            "logs": execution_logs,
            "rag_context": rag_context,
        }


class AuditorAgent:
    def __init__(self, client: genai.Client, model_id: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.model_id = model_id

    def audit(self, issue: str, policies: str, proposed_facts: str) -> dict:
        prompt = f"""
        You are the Red Team Compliance Auditor for the IT Helpdesk.
        Critically evaluate the Researcher's proposed facts against the Enterprise Policies.

        User Issue: {issue}
        Enterprise Policies: {policies}
        Researcher's Proposed Facts: {proposed_facts}

        Output JSON with:
        - "approved": boolean (true only if facts perfectly match policy)
        - "audit_notes": string (brief strict verdict explanation)
        """
        response = _generate_with_fallback(
            self.client,
            self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AuditResult,
                temperature=0.1,
            ),
        )
        return json.loads(response.text)


class CommunicatorAgent:
    def __init__(self, client: genai.Client, model_id: str = "gemini-3.1-flash-lite"):
        self.client = client
        self.model_id = model_id

    def draft_response(
        self,
        ticket_id: str,
        ticket_desc: str,
        urgency: str,
        research_facts: str,
        current_time: str,
    ) -> str:
        prompt = f"""
        You are the Customer Success & Communications Agent for an Enterprise IT Helpdesk.
        Translate the technical facts into a final resolution output.

        Ticket ID: {ticket_id}
        Current Date/Time: {current_time}
        Ticket Description: {ticket_desc}
        Urgency Level: {urgency}
        Technical Facts: {research_facts}

        INSTRUCTIONS:
        1. If facts mention SEV-1/Critical escalation, write a formal Internal Escalation Log.
        2. Otherwise, write a polite professional email explaining the resolution.
        3. Do NOT invent procedures. Include source document citations.
        """
        response = _generate_with_fallback(
            self.client,
            self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4),
        )
        return response.text