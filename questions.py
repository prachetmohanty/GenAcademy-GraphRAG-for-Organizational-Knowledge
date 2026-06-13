"""
10 benchmark questions for the GraphRAG vs Vector RAG comparison, with gold
entity sets used to compute recall.

Each gold set is a list of human-readable entity names (matching the `name`
field in mock_data.py) that a correct, complete answer should mention.
"""

BENCHMARK_QUESTIONS = [
    {
        "id": "q1",
        "question": "Who worked on the SOC2 audit and what tools did they use?",
        "why": "Decision -DECISION_FOR-> Project, then Person -WORKED_ON-> Project, "
               "Project -USED_TOOL-> Tool. Multi-hop relationship question.",
        "gold_entities": [
            "Nina Castellanos", "Devon Park", "AWS Audit Manager", "Okta",
        ],
    },
    {
        "id": "q2",
        "question": "What decisions were made about the pricing model and who approved them?",
        "why": "Decision -DECISION_FOR-> Project, then Person -APPROVED-> Decision.",
        "gold_entities": [
            "Adopt 3-tier usage-based pricing model", "Samira El-Amin", "Priya Subramaniam",
        ],
    },
    {
        "id": "q3",
        "question": "Which project is the Signal Detection Platform's architecture spec "
                     "related to, and who authored it?",
        "why": "Document -REFERENCES-> Project, Person -AUTHORED-> Document.",
        "gold_entities": [
            "Signal Detection Platform (Pharmacovigilance)", "Arjun Mehta", "Leo Fischer",
        ],
    },
    {
        "id": "q4",
        "question": "What skills does Oliver Tan have and which project uses those skills?",
        "why": "Person -HAS_SKILL-> Skill, Project -REQUIRES_SKILL-> Skill.",
        "gold_entities": [
            "Oliver Tan", "Regulatory Affairs (GxP/Part 11)", "Regulatory Submission Assistant",
        ],
    },
    {
        "id": "q5",
        "question": "Who approved the decision to mandate MFA for production systems?",
        "why": "Person -APPROVED-> Decision, plus Document -AUTHORED- link for context.",
        "gold_entities": [
            "Grace Adeyemi", "Mandate MFA for all production system access", "Devon Park",
        ],
    },
    {
        "id": "q6",
        "question": "What tools were standardized for ML training pipelines and who "
                     "proposed that decision?",
        "why": "Decision -DECISION_FOR-> Project -USED_TOOL-> Tool, Person -PROPOSED-> Decision.",
        "gold_entities": [
            "PyTorch", "Standardize on PyTorch for ML training pipelines", "Arjun Mehta",
        ],
    },
    {
        "id": "q7",
        "question": "Which documents reference the Data Quality Remediation project, "
                     "and who worked on it?",
        "why": "Document -REFERENCES-> Project, Person -WORKED_ON-> Project.",
        "gold_entities": [
            "Adverse Event Data Quality Runbook",
            "Data Quality Remediation for Adverse Event Feeds",
            "Leo Fischer", "Priya Subramaniam",
        ],
    },
    {
        "id": "q8",
        "question": "Who is the VP Engineering and which decisions did they approve?",
        "why": "Person attribute lookup + Person -APPROVED-> Decision (multiple edges).",
        "gold_entities": [
            "Grace Adeyemi",
            "Mandate MFA for all production system access",
            "Standardize on PyTorch for ML training pipelines",
        ],
    },
    {
        "id": "q9",
        "question": "What is the status of the Regulatory Submission Assistant project "
                     "and who owns it?",
        "why": "Project attribute lookup + Person -OWNS-> Project.",
        "gold_entities": [
            "Regulatory Submission Assistant", "Priya Subramaniam", "Oliver Tan",
        ],
    },
    {
        "id": "q10",
        "question": "Which security tools are used across projects and who is the "
                     "security engineer?",
        "why": "Project -USED_TOOL-> Tool (security-related) + Person attribute lookup.",
        "gold_entities": [
            "Devon Park", "Okta", "AWS Audit Manager", "Information Security Policy v4",
        ],
    },
]
