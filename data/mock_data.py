"""
Mock organizational knowledge base for Veritask Sciences -- a digital-product /
AI-implementation consultancy building safety, pharmacovigilance, and regulatory
compliance data products for pharma & biotech clients.

This module is the single source of truth for both pipelines:
  - vector_rag.py flattens every node into one text chunk and embeds it.
  - graph_builder.py turns the same nodes + EDGES into a NetworkX MultiDiGraph.

Schema per node: {"id": str, "name": str, "type": str, ...attributes, "text": str}
EDGES: list of (source_id, target_id, relation_type) tuples.
"""

from __future__ import annotations

# ----- People (8) -----
PEOPLE = [
    {"id": "p_nina", "name": "Nina Castellanos", "type": "Person",
     "role": "Compliance Lead",
     "text": "Nina Castellanos is the Compliance Lead at Veritask Sciences, "
             "responsible for GxP and 21 CFR Part 11 readiness across client engagements."},
    {"id": "p_arjun", "name": "Arjun Mehta", "type": "Person",
     "role": "Senior ML Engineer",
     "text": "Arjun Mehta is a Senior ML Engineer at Veritask Sciences, focused on "
             "signal-detection models for pharmacovigilance pipelines."},
    {"id": "p_priya", "name": "Priya Subramaniam", "type": "Person",
     "role": "Product Manager",
     "text": "Priya Subramaniam is a Product Manager at Veritask Sciences, owning the "
             "roadmap for regulatory and data-quality products."},
    {"id": "p_leo", "name": "Leo Fischer", "type": "Person",
     "role": "Data Scientist",
     "text": "Leo Fischer is a Data Scientist at Veritask Sciences, working on adverse "
             "event data quality and signal-detection model evaluation."},
    {"id": "p_sam", "name": "Samira El-Amin", "type": "Person",
     "role": "CFO",
     "text": "Samira El-Amin is the CFO of Veritask Sciences, overseeing pricing "
             "strategy and financial planning."},
    {"id": "p_devon", "name": "Devon Park", "type": "Person",
     "role": "Security Engineer",
     "text": "Devon Park is the Security Engineer at Veritask Sciences, responsible "
             "for access control, audit evidence collection, and security policy."},
    {"id": "p_grace", "name": "Grace Adeyemi", "type": "Person",
     "role": "VP Engineering",
     "text": "Grace Adeyemi is the VP of Engineering at Veritask Sciences, approving "
             "major technical and security decisions."},
    {"id": "p_oliver", "name": "Oliver Tan", "type": "Person",
     "role": "Regulatory Affairs Specialist",
     "text": "Oliver Tan is a Regulatory Affairs Specialist at Veritask Sciences, "
             "supporting submission tooling and GxP documentation."},
]

# ----- Projects (5) -----
PROJECTS = [
    {"id": "proj_pv", "name": "Signal Detection Platform (Pharmacovigilance)",
     "type": "Project", "status": "Active",
     "text": "The Signal Detection Platform is an active pharmacovigilance project "
             "at Veritask Sciences that detects adverse-event signals from client "
             "drug-safety data feeds using machine learning models."},
    {"id": "proj_audit", "name": "SOC2 / Part 11 Compliance Audit Q1 2026",
     "type": "Project", "status": "Completed",
     "text": "The SOC2 / Part 11 Compliance Audit Q1 2026 was a completed project to "
             "prepare Veritask Sciences' systems for SOC2 Type II and 21 CFR Part 11 "
             "audits, including access control review and evidence collection."},
    {"id": "proj_pricing", "name": "Usage-Based Pricing Redesign",
     "type": "Project", "status": "Active",
     "text": "The Usage-Based Pricing Redesign is an active project at Veritask "
             "Sciences to redesign client pricing around a tiered, usage-based model."},
    {"id": "proj_subreg", "name": "Regulatory Submission Assistant",
     "type": "Project", "status": "Active",
     "text": "The Regulatory Submission Assistant is an active project at Veritask "
             "Sciences building AI-assisted tooling to help clients prepare regulatory "
             "submission documents."},
    {"id": "proj_dq", "name": "Data Quality Remediation for Adverse Event Feeds",
     "type": "Project", "status": "Completed",
     "text": "Data Quality Remediation for Adverse Event Feeds was a completed project "
             "at Veritask Sciences that cleaned and validated incoming adverse-event "
             "data feeds used by the Signal Detection Platform."},
]

# ----- Skills (5) -----
SKILLS = [
    {"id": "skill_ml", "name": "ML Engineering", "type": "Skill",
     "text": "ML Engineering: building and maintaining machine learning models and "
             "training pipelines."},
    {"id": "skill_reg", "name": "Regulatory Affairs (GxP/Part 11)", "type": "Skill",
     "text": "Regulatory Affairs (GxP/Part 11): expertise in pharmaceutical regulatory "
             "compliance frameworks including GxP and 21 CFR Part 11."},
    {"id": "skill_de", "name": "Data Engineering", "type": "Skill",
     "text": "Data Engineering: designing and maintaining data pipelines and "
             "data-quality processes."},
    {"id": "skill_sec", "name": "Security & Access Control", "type": "Skill",
     "text": "Security & Access Control: managing identity, access management, and "
             "security policy enforcement."},
    {"id": "skill_prod", "name": "Product Strategy", "type": "Skill",
     "text": "Product Strategy: defining product roadmaps, pricing, and go-to-market "
             "decisions."},
]

# ----- Tools (6) -----
TOOLS = [
    {"id": "tool_pytorch", "name": "PyTorch", "type": "Tool",
     "text": "PyTorch is the machine learning framework standardized for training "
             "pipelines at Veritask Sciences."},
    {"id": "tool_snowflake", "name": "Snowflake", "type": "Tool",
     "text": "Snowflake is the cloud data warehouse used by Veritask Sciences for "
             "adverse-event data storage and analytics."},
    {"id": "tool_audit_mgr", "name": "AWS Audit Manager", "type": "Tool",
     "text": "AWS Audit Manager is the primary cloud control attestation source used "
             "by Veritask Sciences for SOC2 and Part 11 audit evidence."},
    {"id": "tool_okta", "name": "Okta", "type": "Tool",
     "text": "Okta is the identity and access management tool used by Veritask "
             "Sciences for production system authentication, including MFA enforcement."},
    {"id": "tool_langgraph", "name": "LangChain/LangGraph", "type": "Tool",
     "text": "LangChain/LangGraph is the orchestration framework used to build "
             "retrieval-augmented and agentic workflows at Veritask Sciences."},
    {"id": "tool_tableau", "name": "Tableau", "type": "Tool",
     "text": "Tableau is the business intelligence and dashboarding tool used by "
             "Veritask Sciences for reporting on project and pricing metrics."},
]

# ----- Documents (5) -----
DOCUMENTS = [
    {"id": "doc_soc2", "name": "Q1 2026 SOC2 Audit Final Report", "type": "Document",
     "text": "The Q1 2026 SOC2 Audit Final Report documents the results of the SOC2 / "
             "Part 11 Compliance Audit Q1 2026 at Veritask Sciences. It recommends AWS "
             "Audit Manager as the primary cloud control attestation source and notes "
             "that MFA was enforced for all production systems via Okta. Compiled by "
             "Nina Castellanos with security evidence gathered by Devon Park."},
    {"id": "doc_pricing_memo", "name": "Pricing Strategy Memo (March 2026)", "type": "Document",
     "text": "The Pricing Strategy Memo (March 2026) proposes adopting a 3-tier "
             "usage-based pricing model for Veritask Sciences' client contracts. "
             "Authored by Samira El-Amin with input from Priya Subramaniam, and "
             "approved as part of the Usage-Based Pricing Redesign project."},
    {"id": "doc_pv_arch", "name": "Signal Detection Platform Architecture Spec", "type": "Document",
     "text": "The Signal Detection Platform Architecture Spec describes the ML "
             "architecture for the Signal Detection Platform (Pharmacovigilance) "
             "project, including the PyTorch-based training pipeline. Authored by "
             "Arjun Mehta with model-evaluation sections contributed by Leo Fischer."},
    {"id": "doc_dq_runbook", "name": "Adverse Event Data Quality Runbook", "type": "Document",
     "text": "The Adverse Event Data Quality Runbook documents the data-cleaning and "
             "validation steps used in the Data Quality Remediation for Adverse Event "
             "Feeds project, which feeds the Signal Detection Platform. Authored by "
             "Leo Fischer and reviewed by Priya Subramaniam."},
    {"id": "doc_sec_policy", "name": "Information Security Policy v4", "type": "Document",
     "text": "Information Security Policy v4 sets out access-control requirements at "
             "Veritask Sciences, including the mandate for multi-factor authentication "
             "on all production systems via Okta. Maintained by Devon Park."},
]

# ----- Decisions (4) -----
DECISIONS = [
    {"id": "dec_pricing", "name": "Adopt 3-tier usage-based pricing model", "type": "Decision",
     "text": "Decision: Adopt a 3-tier usage-based pricing model for client contracts, "
             "as proposed in the Pricing Strategy Memo (March 2026), part of the "
             "Usage-Based Pricing Redesign project."},
    {"id": "dec_pytorch", "name": "Standardize on PyTorch for ML training pipelines", "type": "Decision",
     "text": "Decision: Standardize on PyTorch for all ML training pipelines at "
             "Veritask Sciences, applied first in the Signal Detection Platform "
             "(Pharmacovigilance) project."},
    {"id": "dec_audit_mgr", "name": "Use AWS Audit Manager as primary cloud control attestation source", "type": "Decision",
     "text": "Decision: Use AWS Audit Manager as the primary cloud control attestation "
             "source for SOC2 and Part 11 audits, made during the SOC2 / Part 11 "
             "Compliance Audit Q1 2026."},
    {"id": "dec_mfa", "name": "Mandate MFA for all production system access", "type": "Decision",
     "text": "Decision: Mandate multi-factor authentication (MFA) for all production "
             "system access via Okta, codified in Information Security Policy v4."},
]

ALL_NODES = PEOPLE + PROJECTS + SKILLS + TOOLS + DOCUMENTS + DECISIONS

# ----- Edges: (source_id, target_id, relation_type) -----
EDGES = [
    # --- WORKED_ON: person -> project ---
    ("p_arjun", "proj_pv", "WORKED_ON"),
    ("p_leo", "proj_pv", "WORKED_ON"),
    ("p_leo", "proj_dq", "WORKED_ON"),
    ("p_priya", "proj_dq", "WORKED_ON"),
    ("p_priya", "proj_subreg", "WORKED_ON"),
    ("p_oliver", "proj_subreg", "WORKED_ON"),
    ("p_nina", "proj_audit", "WORKED_ON"),
    ("p_devon", "proj_audit", "WORKED_ON"),
    ("p_priya", "proj_pricing", "WORKED_ON"),
    ("p_sam", "proj_pricing", "WORKED_ON"),

    # --- USED_TOOL: project -> tool ---
    ("proj_pv", "tool_pytorch", "USED_TOOL"),
    ("proj_pv", "tool_snowflake", "USED_TOOL"),
    ("proj_dq", "tool_snowflake", "USED_TOOL"),
    ("proj_audit", "tool_audit_mgr", "USED_TOOL"),
    ("proj_audit", "tool_okta", "USED_TOOL"),
    ("proj_subreg", "tool_langgraph", "USED_TOOL"),
    ("proj_pricing", "tool_tableau", "USED_TOOL"),

    # --- HAS_SKILL: person -> skill ---
    ("p_arjun", "skill_ml", "HAS_SKILL"),
    ("p_leo", "skill_ml", "HAS_SKILL"),
    ("p_leo", "skill_de", "HAS_SKILL"),
    ("p_nina", "skill_reg", "HAS_SKILL"),
    ("p_oliver", "skill_reg", "HAS_SKILL"),
    ("p_devon", "skill_sec", "HAS_SKILL"),
    ("p_sam", "skill_prod", "HAS_SKILL"),
    ("p_priya", "skill_prod", "HAS_SKILL"),

    # --- REQUIRES_SKILL: project -> skill ---
    ("proj_subreg", "skill_reg", "REQUIRES_SKILL"),
    ("proj_pv", "skill_ml", "REQUIRES_SKILL"),
    ("proj_dq", "skill_de", "REQUIRES_SKILL"),
    ("proj_audit", "skill_sec", "REQUIRES_SKILL"),
    ("proj_pricing", "skill_prod", "REQUIRES_SKILL"),

    # --- AUTHORED: person -> document ---
    ("p_nina", "doc_soc2", "AUTHORED"),
    ("p_devon", "doc_soc2", "AUTHORED"),
    ("p_sam", "doc_pricing_memo", "AUTHORED"),
    ("p_priya", "doc_pricing_memo", "AUTHORED"),
    ("p_arjun", "doc_pv_arch", "AUTHORED"),
    ("p_leo", "doc_pv_arch", "AUTHORED"),
    ("p_leo", "doc_dq_runbook", "AUTHORED"),
    ("p_priya", "doc_dq_runbook", "AUTHORED"),
    ("p_devon", "doc_sec_policy", "AUTHORED"),

    # --- REFERENCES: document -> project / decision ---
    ("doc_soc2", "proj_audit", "REFERENCES"),
    ("doc_pricing_memo", "proj_pricing", "REFERENCES"),
    ("doc_pv_arch", "proj_pv", "REFERENCES"),
    ("doc_dq_runbook", "proj_dq", "REFERENCES"),
    ("doc_dq_runbook", "proj_pv", "REFERENCES"),
    ("doc_sec_policy", "dec_mfa", "REFERENCES"),
    ("doc_soc2", "dec_audit_mgr", "REFERENCES"),
    ("doc_pricing_memo", "dec_pricing", "REFERENCES"),

    # --- DECISION_FOR: decision -> project ---
    ("dec_pricing", "proj_pricing", "DECISION_FOR"),
    ("dec_pytorch", "proj_pv", "DECISION_FOR"),
    ("dec_audit_mgr", "proj_audit", "DECISION_FOR"),

    # --- APPROVED: person -> decision ---
    ("p_sam", "dec_pricing", "APPROVED"),
    ("p_priya", "dec_pricing", "APPROVED"),
    ("p_grace", "dec_pytorch", "APPROVED"),
    ("p_grace", "dec_mfa", "APPROVED"),
    ("p_nina", "dec_audit_mgr", "APPROVED"),
    ("p_grace", "dec_audit_mgr", "APPROVED"),

    # --- PROPOSED: person -> decision ---
    ("p_arjun", "dec_pytorch", "PROPOSED"),

    # --- OWNS: person -> project ---
    ("p_priya", "proj_subreg", "OWNS"),
    ("p_priya", "proj_pricing", "OWNS"),
    ("p_oliver", "proj_subreg", "OWNS"),
    ("p_arjun", "proj_pv", "OWNS"),
    ("p_nina", "proj_audit", "OWNS"),

    # --- DECISION_FOR: decision -> document (MFA mandate -> sec policy) ---
    ("dec_mfa", "doc_sec_policy", "DECISION_FOR"),
]
