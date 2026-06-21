import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

# --- Pydantic Models for Structured Output ---
class TicketAnalysis(BaseModel):
    urgency: str
    category: str
    summary: str

class AuditResult(BaseModel):
    approved: bool
    audit_notes: str

# ==========================================
# AGENT 1: TRIAGE AGENT (The Receptionist)
# ==========================================
class TriageAgent:
    def __init__(self, client: genai.Client, model_id: str = 'gemini-2.5-flash-lite'):
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
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TicketAnalysis,
                temperature=0.1
            ),
        )
        return json.loads(response.text)

# ==========================================
# AGENT 2: RESEARCHER AGENT (Tier 3 Engineer w/ Sandbox)
# ==========================================
class ResearcherAgent:
    def __init__(self, client: genai.Client, rag_mcp, model_id: str = 'gemini-2.5-flash-lite'):
        self.client = client
        self.rag_mcp = rag_mcp
        self.model_id = model_id

    def investigate(self, issue_summary: str) -> dict:
        rag_context = self.rag_mcp.search_knowledge_base(issue_summary)
        
        prompt = f"""
        You are an expert Tier 3 IT Security & Research Engineer.
        Your job is to read the enterprise policy context provided and extract ONLY the exact technical steps or rules that apply to the user's issue.
        
        CRITICAL INSTRUCTIONS:
        1. You are strictly FORBIDDEN from calculating manually or predicting math. If the user issue contains numbers, arrays, or logs, you MUST use the code_execution tool to write and run a Python script to calculate the exact answer. You must wait for the output before generating your final text.
        2. You MUST explicitly cite the document name (e.g., "According to [SOURCE DOCUMENT: filename]") for every rule you extract.
        3. Do NOT write an email. Do NOT be polite. Just state the facts, protocols, and mathematical results.

        User Issue: {issue_summary}
        
        Enterprise Policy Context:
        {rag_context}
        """
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                tools=[{"code_execution": {}}]
            )
        )
        
        execution_logs = []
        final_text = ""
        
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.executable_code:
                    execution_logs.append({"type": "code", "content": part.executable_code.code})
                elif part.code_execution_result:
                    execution_logs.append({"type": "output", "content": part.code_execution_result.output})
                elif part.text:
                    final_text += part.text
                    
        return {
            "text": final_text.strip(),
            "logs": execution_logs,
            "rag_context": rag_context # Return context so the Auditor can read it
        }

# ==========================================
# AGENT 3: RED TEAM AUDITOR (The Quality Control)
# ==========================================
class AuditorAgent:
    def __init__(self, client: genai.Client, model_id: str = 'gemini-2.5-flash-lite'):
        self.client = client
        self.model_id = model_id

    def audit(self, issue: str, policies: str, proposed_facts: str) -> dict:
        prompt = f"""
        You are the Red Team Compliance Auditor for the IT Helpdesk.
        Your job is to critically evaluate the Researcher Agent's proposed facts against the Enterprise Policies.
        
        User Issue: {issue}
        Enterprise Policies: {policies}
        Researcher's Proposed Facts: {proposed_facts}
        
        INSTRUCTIONS:
        1. Evaluate if the Researcher missed any policy rule or made a wrong escalation.
        2. Output JSON with:
           - "approved": boolean (true if facts perfectly match policy, false if violation).
           - "audit_notes": string (Briefly explain your verdict. Be strict).
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AuditResult,
                temperature=0.1
            )
        )
        return json.loads(response.text)

# ==========================================
# AGENT 4: COMMUNICATOR AGENT (PR & Success)
# ==========================================
class CommunicatorAgent:
    def __init__(self, client: genai.Client, model_id: str = 'gemini-2.5-flash-lite'):
        self.client = client
        self.model_id = model_id

    def draft_response(self, ticket_id: str, ticket_desc: str, urgency: str, research_facts: str, current_time: str) -> str:
        prompt = f"""
        You are the Customer Success & Communications Agent for an Enterprise IT Helpdesk.
        Your job is to translate cold technical facts into a final resolution output.

        Ticket ID: {ticket_id}
        Current Date/Time: {current_time}
        Ticket Description: {ticket_desc}
        Urgency Level: {urgency}
        Technical Facts to enforce: {research_facts}

        INSTRUCTIONS:
        1. If the Technical Facts mention a SEV-1/Critical escalation, write a formal Internal Escalation Log.
        2. If it is a standard user request, write a polite, professional email explaining the rules.
        3. Do NOT invent new procedures. Include the source document citations.
        """
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4)
        )
        return response.text