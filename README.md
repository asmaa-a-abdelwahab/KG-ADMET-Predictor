# KG-ADMET-Predictor


```
/project_folder_structure/
│
├── /neo4j/                # Neo4j-specific files
│   ├── Dockerfile         # Dockerfile for Neo4j
│   └── neo4j.conf         # Neo4j configuration file
│   └── /utils/            # Utility scripts
│
├── /app/                  # Streamlit app-specific files
│   ├── Dockerfile         # Dockerfile for the Streamlit app
│   ├── app.py             # Streamlit app code
│   ├── requirements.txt   # Python dependencies for the Streamlit app
│   ├── /models/           # Directory for deep learning model files
│   └── /utils/            # Utility scripts
│
├── docker-compose.yml     # Docker Compose configuration
├── LICENSE                # license
└── README.md              # Instructions/documentation

```

## Memory Recommendations for Neo4j:
- Heap Memory (dbms.memory.heap.initial_size and dbms.memory.heap.max_size):

    - Set to 8 GB (8G) for a 16 GB machine, leaving enough memory for other processes.
- Page Cache (dbms.memory.pagecache.size):

    - Set to around 6-7 GB to optimize the memory used for caching the graph in memory.
- Garbage Collection (dbms.jvm.additional=-XX:+UseG1GC):

    - G1 Garbage Collector is often recommended for large heaps in Neo4j.