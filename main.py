from fastapi import FastAPI
from pydantic import BaseModel
import networkx as nx

# Import from our custom modules
from schemas import InvestigationResult, FactCheckResult
from rag import corpus
from agents import run_investigator, run_fact_checker, interrogate_suspect

app = FastAPI(title="180DC Investigation Backend")

# --- Request Models ---
class InterrogateRequest(BaseModel):
    suspect_name: str
    question: str

class SearchRequest(BaseModel):
    query: str

class InvestigateRequest(BaseModel):
    query: str

class FactCheckRequest(BaseModel):
    theory: str

class VerdictRequest(BaseModel):
    suspect: str
    justification: str

# --- Endpoints ---
@app.post("/ingest")
def ingest_data():
    """Processes the selected corpus and returns the evidence graph."""
    corpus.ingest_corpus()
    nodes = list(corpus.graph.nodes(data=True))
    edges = list(corpus.graph.edges(data=True))
    return {"status": "success", "message": "Corpus and graph generated.", "graph": {"nodes": nodes, "edges": edges}}

@app.post("/interrogate")
def interrogate(req: InterrogateRequest):
    """Chat with a candidate/suspect."""
    response = interrogate_suspect(req.suspect_name, req.question)
    return {"response": response}

@app.post("/search_evidence")
def search_evidence(req: SearchRequest):
    """Search the evidence corpus."""
    results = corpus.retrieve(req.query, top_k=3)
    return {"results": results}

@app.post("/investigate")
def investigate(req: InvestigateRequest):
    """Run the Investigator Agent."""
    result = run_investigator(req.query)
    return result

@app.post("/fact_check")
def fact_check(req: FactCheckRequest):
    """Challenge the Investigator's theory."""
    result = run_fact_checker(req.theory)
    return result

@app.post("/submit_verdict")
def submit_verdict(req: VerdictRequest):
    """Evaluate the user's final conclusion."""
    # Simple hardcoded logic to evaluate if they fell for the misleading evidence
    if "alice" in req.suspect.lower():
        evaluation = "FAILED: You relied on the unverified anonymous tip. Alice's diner receipt clears her."
    elif "bob" in req.suspect.lower():
        evaluation = "SUCCESS: Bob has gambling debts, and his 'restroom' alibi perfectly matches the camera outage time."
    else:
        evaluation = "INCONCLUSIVE: The evidence strongly points to Bob."
        
    return {"evaluation": evaluation, "your_suspect": req.suspect, "your_justification": req.justification}