import os
import json
import csv
import io
import time
from datetime import datetime
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from agents.crew import TriageAgent, ResearcherAgent, AuditorAgent, CommunicatorAgent
from mcp_server.rag_mcp import DocumentRAG
from tools.security import SecurityGuard

st.set_page_config(page_title="Enterprise AI Helpdesk", page_icon="🏢", layout="wide")

# --- PATHS ---
base_path = Path(__file__).parent.parent
tickets_path = base_path / "data" / "tickets.json"
history_path = base_path / "data" / "history.json"

# --- HELPER: REAL METRICS TRACKING ---
def log_ticket_resolution(ticket_id, category, urgency, security_blocks):
    history = []
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            try: history = json.load(f)
            except: history = []
            
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticket_id": ticket_id,
        "category": category,
        "urgency": urgency,
        "security_blocks": security_blocks
    })
    
    os.makedirs(history_path.parent, exist_ok=True)
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

# --- SESSION STATE INITIALIZATION ---
if 'agent_thoughts' not in st.session_state: st.session_state.agent_thoughts = []
if 'pipeline_stage' not in st.session_state: st.session_state.pipeline_stage = "idle" 
if 'active_ticket_id' not in st.session_state: st.session_state.active_ticket_id = None
if 'active_description' not in st.session_state: st.session_state.active_description = ""
if 'triage_data' not in st.session_state: st.session_state.triage_data = {}
if 'research_facts' not in st.session_state: st.session_state.research_facts = ""
if 'research_logs' not in st.session_state: st.session_state.research_logs = []
if 'audit_data' not in st.session_state: st.session_state.audit_data = {}
if 'final_resolution' not in st.session_state: st.session_state.final_resolution = ""
if 'active_interventions' not in st.session_state: st.session_state.active_interventions = []
if 'show_db_explorer' not in st.session_state: st.session_state.show_db_explorer = False

def add_thought(agent_name, thought):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_thoughts.append(f"**[{timestamp}] {agent_name}**: {thought}")

# --- SYSTEM LOAD (CACHED) ---
@st.cache_resource
def load_system():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("⚠️ GEMINI_API_KEY not found in .env file")
        
    client = genai.Client(api_key=api_key)
    mcp = DocumentRAG()
    security = SecurityGuard()
    triage = TriageAgent(client=client)
    researcher = ResearcherAgent(client=client, rag_mcp=mcp)
    auditor = AuditorAgent(client=client)
    communicator = CommunicatorAgent(client=client)
    
    return mcp, security, triage, researcher, auditor, communicator

try:
    mcp, security, triage, researcher, auditor, communicator = load_system()
except Exception as e:
    st.error(f"⚠️ Error initializing system components: {e}")
    st.stop()

# --- METRICS CALCULATION ---
total_resolved = len(json.load(open(history_path, 'r'))) if history_path.exists() else 0
total_security_blocks = sum(item.get("security_blocks", 0) for item in json.load(open(history_path, 'r'))) if history_path.exists() else 0

# --- SIDEBAR: MONITORING & LIVE CHAIN OF THOUGHT ---
with st.sidebar:
    st.header("📊 Dashboard Metrics")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Resolved", total_resolved)
    col_m2.metric("DLP Blocks", total_security_blocks)
    st.divider()
    
    st.header("🧠 Agent Reasoning")
    if st.session_state.agent_thoughts:
        for thought in reversed(st.session_state.agent_thoughts):
            st.markdown(thought)
    else: st.caption("Waiting for active pipeline actions...")
    
    st.divider()
    st.success(f"Vector Database Chunks: {mcp.collection.count()}")
    if st.button("Reset Pipeline", type="secondary"):
        st.session_state.pipeline_stage = "idle"
        st.session_state.agent_thoughts = []
        st.rerun()

# --- MAIN PAGE HEADERS ---
st.title("Enterprise AI Helpdesk Command Center")
st.markdown("Automated Multi-Agent Pipeline with **Debate, Sandbox Validation & Source Attribution**")
st.divider()

# --- PIPELINE ACTIONS ---
def run_initial_analysis(ticket_id, description):
    st.session_state.agent_thoughts = []
    st.session_state.active_ticket_id = ticket_id
    st.session_state.active_description = description
    st.session_state.show_db_explorer = False
    
    # VISUAL FIX: Added st.write and time.sleep for sequential animation.
    with st.status(f"⚙️ Orchestrating Multi-Agent Swarm for {ticket_id}...", expanded=True) as status:
        
        st.write("🕵️‍♂️ **Triage Agent:** Extracting metadata and priority matrix...")
        add_thought("Triage Agent", "Scanning textual structure to classify category.")
        st.session_state.triage_data = triage.analyze(description)
        time.sleep(1.5) # Visual pause
        st.write(f"✅ *Classification Complete: {st.session_state.triage_data.get('category')}*")
        
        st.write("🔬 **Researcher Agent:** Querying Vector DB & executing Sandbox...")
        add_thought("Researcher Agent", "Initiating Semantic Query against Vector Knowledge Base.")
        research_result = researcher.investigate(st.session_state.triage_data.get('summary', description))
        st.session_state.research_facts = research_result["text"]
        st.session_state.research_logs = research_result.get("logs", [])
        time.sleep(2) # Visual pause
        st.write(f"✅ *Fact Extraction Complete. Computations executed: {len(st.session_state.research_logs)}*")
        
        st.write("⚖️ **Red Team Auditor:** Cross-referencing findings with policy constraints...")
        add_thought("Auditor Agent", "Critically evaluating Researcher's proposed facts.")
        st.session_state.audit_data = auditor.audit(
            issue=st.session_state.triage_data.get('summary', description),
            policies=research_result["rag_context"],
            proposed_facts=research_result["text"]
        )
        time.sleep(1.5) # Visual pause
        
        if st.session_state.audit_data.get("approved"):
            st.write("✅ *Audit Passed: 100% Policy Compliance.*")
            add_thought("Auditor Agent", "Facts verified. No policy violations found.")
        else:
            st.write("❌ *Audit Warning: Policy deviations detected.*")
            add_thought("Auditor Agent", "WARNING: Found inconsistencies with corporate policies.")
            
        st.session_state.pipeline_stage = "awaiting_review"
        status.update(label="Initial Swarm Phase Complete. Awaiting Human Gate.", state="complete")
    
    time.sleep(1) # Final pause before rerun
    st.rerun()

def run_final_generation(edited_facts):
    with st.status("✍️ Compiling Final Delivery...", expanded=True) as status:
        add_thought("Communicator Agent", "Assembling professional output format.")
        raw_res = communicator.draft_response(
            st.session_state.active_ticket_id, st.session_state.active_description,
            st.session_state.triage_data.get('urgency', 'Low'), edited_facts, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        add_thought("Security Guard", "Inspecting final draft for PII and infrastructure leaks (DLP scan).")
        security_report = security.sanitize_output(raw_res)
        st.session_state.final_resolution = security_report["safe_text"]
        
        interventions = security_report["interventions"]
        st.session_state.active_interventions = interventions
        log_ticket_resolution(st.session_state.active_ticket_id, st.session_state.triage_data.get('category'), st.session_state.triage_data.get('urgency'), len(interventions) if interventions else 0)
        
        st.session_state.pipeline_stage = "completed"
        status.update(label="Workflow Engine Finished Processing.", state="complete")
    st.rerun()

# --- INTERFACE TABS ---
tab_queue, tab_new, tab_admin = st.tabs(["📋 Pending Tickets", "📥 Submit Live Ticket", "⚙️ Admin & Compliance"])

with tab_queue:
    if st.session_state.pipeline_stage == "idle":
        try: tickets = json.load(open(tickets_path, 'r', encoding='utf-8'))
        except: tickets = []
        for ticket in tickets:
            with st.expander(f"ID: {ticket['ticket_id']} | Subject: {ticket['subject']}"):
                st.write(f"**Description:** {ticket['description']}")
                if st.button("Launch Pipeline Analysis", key=f"btn_{ticket['ticket_id']}"):
                    run_initial_analysis(ticket['ticket_id'], ticket['description'])
    else: st.info(f"🔒 Queue locked. Processing ticket: **{st.session_state.active_ticket_id}**")

with tab_new:
    if st.session_state.pipeline_stage == "idle":
        with st.form("new_ticket_submission_form"):
            new_description = st.text_area("Full Raw Log / Incident Details Description", height=150)
            if st.form_submit_button("Submit & Start Automated Pipeline") and new_description:
                run_initial_analysis("LIVE-TKT-999", new_description)

with tab_admin:
    col_upload, col_stats = st.columns(2)
    with col_upload:
        st.subheader("📚 Knowledge Base Ingestion")
        uploaded_file = st.file_uploader("Upload Policy Document", type=['txt', 'md'])
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue().decode("utf-8")
            if st.button("Process Document into Vector DB"):
                chunks_added = mcp.ingest_new_document(file_content, uploaded_file.name)
                st.success(f"Successfully digested {chunks_added} chunks from `{uploaded_file.name}`.")
                st.rerun()
                
        # --- VECTOR DATABASE EXPLORER ---
        st.divider()
        st.subheader("🗄️ Vector Database Explorer")
        st.caption("Visually inspect the embeddings and chunks currently stored in ChromaDB.")
        
        if st.button("👁️ View Database Chunks"):
            st.session_state.show_db_explorer = not st.session_state.show_db_explorer
            
        if st.session_state.show_db_explorer:
            db_contents = mcp.collection.get()
            if db_contents and db_contents.get("ids"):
                st.info(f"Loaded {len(db_contents['ids'])} knowledge chunks from Vector Storage.")
                for i in range(len(db_contents['ids'])):
                    chunk_id = db_contents['ids'][i]
                    doc_text = db_contents['documents'][i]
                    meta = db_contents['metadatas'][i]
                    source = meta.get('source', 'Unknown') if meta else 'Unknown'
                    
                    with st.expander(f"📄 {source} (ID: {chunk_id[:15]}...)", expanded=False):
                        st.markdown("**Content Data:**")
                        st.write(doc_text)
                        st.caption(f"Full Hash ID: `{chunk_id}`")
            else:
                st.warning("Database is currently empty.")
                
    with col_stats:
        st.subheader("📊 Compliance Audit & Reporting Export")
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                try:
                    h_data = json.load(f)
                except:
                    h_data = []
            if h_data:
                st.dataframe(h_data)
                
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=["timestamp", "ticket_id", "category", "urgency", "security_blocks"])
                writer.writeheader()
                writer.writerows(h_data)
                csv_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Certified Corporate Audit Log (CSV)",
                    data=csv_data,
                    file_name=f"compliance_audit_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.caption("No historical executions recorded yet.")
        else:
            st.caption("No corporate resolution footprint found.")

# ==========================================================
# THE INTERACTIVE REVIEW GATEWAY (HUMAN-IN-THE-LOOP)
# ==========================================================
if st.session_state.pipeline_stage != "idle":
    st.divider()
    st.header(f"🎛️ Human-in-the-Loop Gateway: {st.session_state.active_ticket_id}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Categorization Vector", st.session_state.triage_data.get('category', 'Unknown'))
    col2.metric("Assigned Urgency", st.session_state.triage_data.get('urgency', 'Low'))
    col3.info(f"**AI Summary:**\n{st.session_state.triage_data.get('summary', '')}")
    
    if st.session_state.pipeline_stage == "awaiting_review":
        st.subheader("📋 Step 2: Context Verification & Source Accountability")
        
        # NEW: AUDITOR PANEL (RED TEAM)
        audit = st.session_state.audit_data
        if audit.get("approved"):
            st.success(f"⚖️ **Red Team Auditor Verdict: APPROVED.** \n\n*Notes:* {audit.get('audit_notes')}")
        else:
            st.error(f"⚖️ **Red Team Auditor Verdict: CHALLENGED.** \n\n*Notes:* {audit.get('audit_notes')}")
            st.warning("⚠️ The Auditor detected that the Researcher Agent's findings might violate corporate policy. Please review the facts below carefully.")
        
        if st.session_state.research_logs:
            st.markdown("#### 🛠️ Live AI Sandbox Runtime Environment")
            for block in st.session_state.research_logs:
                if block["type"] == "code":
                    st.markdown("##### 📝 Autonomously Generated Logic (Python)")
                    st.code(block["content"], language="python")
                elif block["type"] == "output":
                    st.markdown("##### 🖥️ Compute Console Output")
                    st.code(block["content"], language="text")
            st.divider()
            
        human_edited_facts = st.text_area("Retrieved Fact Substrate (Editable)", value=st.session_state.research_facts, height=250)
        
        if st.button("Authorize & Compile Final Resolution", type="primary"):
            run_final_generation(human_edited_facts)
            
    if st.session_state.pipeline_stage == "completed":
        st.subheader("✅ Step 3: Final Safe Delivery Payload")
        if st.session_state.active_interventions:
            for alert in st.session_state.active_interventions:
                st.warning(f"🚨 **DLP Active Shield Triggered:** {alert}")
        else:
            st.success("🔒 **DLP System Clearance**: Zero corporate leaks.")
            
        st.markdown(st.session_state.final_resolution)
        if st.button("Reset & Return", type="secondary"):
            st.session_state.pipeline_stage = "idle"
            st.rerun()