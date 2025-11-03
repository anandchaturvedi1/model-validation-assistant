# Example LangGraph visualization builder (pseudo-code)
def build_example_graph():
    graph = {
        'nodes': [
            {'id':'upload','label':'Upload Doc'},
            {'id':'parser','label':'Parser (PDF/SAS/Python)'},
            {'id':'embed','label':'Embeddings / FAISS'},
            {'id':'llm','label':'LLM (RAG)'},
            {'id':'report','label':'Validation Report'}
        ],
        'edges':[
            ('upload','parser'),
            ('parser','embed'),
            ('embed','llm'),
            ('llm','report')
        ]
    }
    return graph

if __name__ == '__main__':
    g = build_example_graph()
    import json
    print(json.dumps(g, indent=2))