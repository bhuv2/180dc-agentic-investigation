import streamlit as st
import requests
import streamlit.components.v1 as components
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="180DC Investigation Agent", layout="wide")
st.title("🕵️ Interactive Agentic RAG Investigation")

# --- Initialize System ---
if st.button("Initialize / Ingest Corpus"):
    with st.spinner("Building evidence graph and RAG indices..."):
        resp = requests.post(f"{API_URL}/ingest")
        if resp.status_code == 200:
            st.success("Corpus ingested successfully! Evidence graph ready.")
        else:
            st.error("Failed to ingest corpus.")

# --- UI Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Interrogate", "Search Evidence", "Agents (Investigator & Fact-Checker)", "Verdict", "Visual Board"])

# Tab 1: Interrogate
# Replace the contents of "with tab1:" with this:
with tab1:
    st.header("Interrogate a Suspect")
    suspect_name = st.selectbox("Choose Suspect:", ["Alice", "Bob"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if st.button("Clear History"):
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Interrogate..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Send to API (you might want to update your backend to accept history, 
        # but for an MVP, just appending the previous context to the prompt in agents.py works too)
        resp = requests.post(f"{API_URL}/interrogate", json={"suspect_name": suspect_name, "question": prompt})
        reply = resp.json().get('response')
        
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

# Tab 2: Search
with tab2:
    st.header("Search the Evidence Database")
    search_query = st.text_input("Enter keywords or semantic query (e.g., 'timeline of the robbery'):")
    if st.button("Search"):
        if search_query:
            resp = requests.post(f"{API_URL}/search_evidence", json={"query": search_query})
            results = resp.json().get("results", [])
            for res in results:
                st.write(f"**Doc ID:** {res['id']}")
                st.write(res['content'])
                st.divider()

# Tab 3: Agents
with tab3:
    st.header("Agent Investigation Board")
    investigation_query = st.text_input("What should the Investigator look into?", value="Who stole the Diamond Crown?")
    
    if st.button("Run Investigator"):
        with st.spinner("Investigator is forming a theory and retrieving evidence..."):
            resp = requests.post(f"{API_URL}/investigate", json={"query": investigation_query})
            data = resp.json()
            
            st.subheader("Investigator's Theory")
            st.write(data.get("theory"))
            st.write(f"**Confidence:** {data.get('confidence')}")
            
            st.subheader("Citations")
            for cit in data.get("citations", []):
                st.caption(f"- [{cit['document_id']}] {cit['claim']}")
            with st.expander("Show Agent Execution Trace"):
                for step in data.get("execution_trace", []):
                    st.write(f"{step}")
            # Save theory to session state for the Fact-Checker
            st.session_state['current_theory'] = data.get("theory")

    st.divider()
    st.header("Adversarial Fact-Checker")
    if st.button("Run Fact-Checker"):
        if 'current_theory' in st.session_state:
            with st.spinner("Fact-Checker is hunting for contradictions..."):
                resp = requests.post(f"{API_URL}/fact_check", json={"theory": st.session_state['current_theory']})
                data = resp.json()
                
                if data.get("contradictions_found"):
                    st.error("🚨 Contradictions Found!")
                else:
                    st.success("✅ Theory appears solid.")
                st.write(data.get("details"))
        else:
            st.warning("Please run the Investigator first.")

# Tab 4: Verdict
with tab4:
    st.header("Submit Final Verdict")
    final_suspect = st.text_input("Who is the prime suspect?")
    justification = st.text_area("Justification (Reference your evidence):")
    if st.button("Submit Verdict"):
        if final_suspect and justification:
            resp = requests.post(f"{API_URL}/submit_verdict", json={"suspect": final_suspect, "justification": justification})
            data = resp.json()
            st.subheader("Evaluation Feedback")
            st.write(data.get("evaluation"))

# Tab 5: Visual Board
with tab5:
    st.header("Interactive Evidence Graph")
    if st.button("Generate Visual Graph"):
        resp = requests.post(f"{API_URL}/ingest") # Re-fetch graph data
        if resp.status_code == 200:
            graph_data = resp.json().get("graph", {})
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])
            
            # Build Pyvis network
            from pyvis.network import Network
            net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
            
            for node in nodes:
                net.add_node(node[0], label=str(node[0]), color="#00ffcc" if node[1].get("type") == "document" else "#ff0066")
            
            for edge in edges:
                net.add_edge(edge[0], edge[1], title=edge[2].get("relation", ""))
                
            # Save and render
            net.save_graph("evidence_graph.html")
            HtmlFile = open("evidence_graph.html", 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=550)