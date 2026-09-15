import os
import json
import google.generativeai as genai
from schemas import InvestigationResult, FactCheckResult
from rag import corpus

# Configure the Gemini API key from your environment
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# We use gemini-1.5-flash for speed and force the output to be JSON
agent_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json"
    }
)

def run_investigator(query: str, max_retries: int = 2) -> dict:
    """
    Forms an initial theory, checks evidence sufficiency, and retries if needed.
    """
    current_query = query
    trace = [f"Initial query: {query}"] # Initialize trace
    
    for attempt in range(max_retries):
        trace.append(f"Attempt {attempt + 1}: Retrieving docs for '{current_query}'")
        # 1. Retrieve evidence
        retrieved_docs = corpus.retrieve(current_query, top_k=3)
        context = "\n".join([f"Document ID: {doc['id']} | Content: {doc['content']}" for doc in retrieved_docs])
        
        # 2. Formulate the prompt with strict grounding instructions
        prompt = f"""
        You are the Investigator Agent solving a case.
        Analyze the situation regarding: '{query}'.
        
        Retrieved Evidence:
        {context}
        
        CRITICAL INSTRUCTIONS:
        - Be highly suspicious of unverified tips or anonymous sources. Do not treat them as established fact.
        - Cite your sources using the exact Document ID provided.
        - If the evidence is insufficient to make a confident claim, set 'needs_more_evidence' to true.
        - Output your response strictly as JSON matching this schema:
          {{
            "theory": "Your detailed theory here",
            "confidence": 0.0 to 1.0 (float),
            "citations": [{{"document_id": "exact_id", "claim": "exact_claim"}}],
            "needs_more_evidence": true or false
          }}
        """
        
        # 3. Call Gemini
        response = agent_model.generate_content(prompt)
        
        try:
            result_json = json.loads(response.text)
            
            # 4. Agentic Self-Correction: Retry if more evidence is needed
            if result_json.get("needs_more_evidence") and attempt < max_retries - 1:
                trace.append("Agent evaluated evidence as INSUFFICIENT. Reformulating query.")
                print(f"Investigator needs more evidence. Retrying (Attempt {attempt + 1})...")
                # Expand the query for the next RAG pass
                current_query = f"{query} timeline alibi details" 
                continue
            
            trace.append(f"Agent finalized theory with confidence {result_json.get('confidence')}.")
            result_json["execution_trace"] = trace # Attach trace to output
            return result_json
            
        except Exception as e:
            return {"theory": "Error parsing JSON", "confidence": 0.0, "citations": [], "needs_more_evidence": False, "execution_trace": trace}

def run_fact_checker(investigator_theory: str) -> dict:
    """
    Adversarial agent looking for contradictions or alternative timelines.
    """
    retrieved_docs = corpus.retrieve(investigator_theory, top_k=4)
    context = "\n".join([f"Document ID: {doc['id']} | Content: {doc['content']}" for doc in retrieved_docs])
    
    prompt = f"""
    You are the Fact-Checker Agent. Your sole job is to scrutinize the Investigator's theory and find contradictions, alternative explanations, or conflicting timelines.
    
    Investigator's Theory:
    {investigator_theory}
    
    Available Evidence:
    {context}
    
    CRITICAL INSTRUCTIONS:
    - Actively look for alibis or unverified tips that the Investigator might have relied on incorrectly.
    - Output your response strictly as JSON matching this schema:
      {{
        "contradictions_found": true or false,
        "details": "Explanation of any contradictions, weaknesses, or confirmation that the theory is solid."
      }}
    """
    
    response = agent_model.generate_content(prompt)
    try:
         return json.loads(response.text)
    except Exception as e:
         return {"contradictions_found": False, "details": "Failed to parse output."}

def interrogate_suspect(suspect_name: str, question: str) -> str:
    """
    Allows the user to freely chat with a candidate.
    """
    # Find the specific suspect's document for context
    suspect_doc = next((doc for doc in corpus.documents if suspect_name.lower() in doc['id'].lower()), None)
    context = f"Document: {suspect_doc['content']}" if suspect_doc else "No specific background found."
    
    prompt = f"""
    You are playing the role of {suspect_name} in an investigation.
    Here is your background info: {context}
    
    The user (a detective) asks you: "{question}"
    
    Respond in character. Be brief. If the question is outside your background, be evasive. Do NOT output JSON, just plain text.
    """
    
    # Standard text model (no JSON enforcement)
    chat_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = chat_model.generate_content(prompt)
    return response.text