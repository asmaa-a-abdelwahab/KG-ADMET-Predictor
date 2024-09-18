# KG-ADMET-Predictor


```
/project_folder_structure/
│
├── /neo4j/               # Neo4j-specific files
│   ├── Dockerfile         # Dockerfile for Neo4j
│   └── neo4j.conf         # Neo4j configuration file
│
├── /streamlit-app/        # Streamlit app-specific files
│   ├── Dockerfile         # Dockerfile for the Streamlit app
│   ├── app.py             # Streamlit app code
│   ├── requirements.txt   # Python dependencies for the Streamlit app
│   ├── /models/           # Directory for deep learning model files
│   └── /utils/            # Utility scripts
│
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # Instructions/documentation

```