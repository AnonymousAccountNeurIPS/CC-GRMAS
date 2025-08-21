# CC-GRMAS: A Multi-Agent Graph Neural System for Spatiotemporal Landslide Risk Assessment in High Mountain Asia

[![NeurIPS 2025 Workshop](https://img.shields.io/badge/NeurIPS%202025-Climate%20Change%20Workshop-green)](https://neurips.cc/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

CC-GRMAS (Climate Change Graph Risk Management and Analysis System) is a novel multi-agent framework that leverages advanced AI techniques for proactive landslide risk management in High Mountain Asia (HMA). The system combines Graph Neural Networks (GNNs) with attention mechanisms, Retrieval Augmented Generation (RAG), and automated response coordination to provide real-time landslide risk assessment and disaster response planning.

<img width="2308" height="847" alt="architecture" src="https://github.com/user-attachments/assets/f9b5bc7d-567c-4643-bdb1-d918198d6a89" />

## Key Features

- **Multi-Agent Architecture**: Three specialized agents (Prediction, Planning, Execution) working collaboratively
- **Graph Neural Networks**: Spatial relationship modeling with 99.9% parameter reduction compared to traditional CV approaches
- **Real-time Assessment**: Proactive hotspot detection and intervention capabilities
- **Scalable Design**: Modular framework adaptable to various geographies and hazards
- **Climate Impact Integration**: Supports UN Sustainable Development Goals 13, 11, and 15

## System Architecture

The CC-GRMAS framework employs three interlinked agents:

### 1. Prediction Agent
- **Technology**: Graph Neural Networks with attention mechanisms
- **Function**: Models spatial relationships between landslide events
- **Performance**: F1-score of 0.7981 with only 42.7k parameters
- **Features**: 
  - Spatial coordinate normalization
  - Temporal encoding
  - Impact severity quantification
  - Dynamic proximity graph construction

### 2. Planning Agent
- **Technology**: Large Language Models with Retrieval Augmented Generation
- **Function**: Context-aware risk analysis and climate impact assessments
- **Features**:
  - Graph-based knowledge integration
  - Vector embeddings for semantic search
  - Neo4j graph database integration
  - Domain-specific prompt engineering

### 3. Execution Agent
- **Technology**: Automated response coordination workflows
- **Function**: Operational response generation and hotspot detection
- **Features**:
  - Grid-based spatial sampling
  - Automated alert generation
  - Response recommendation system

## Dataset

The system utilizes the **NASA Global Landslide Catalog (GLC)** with:
- **1,558 landslide events** in High Mountain Asia
- **Time period**: 2007-2020
- **Sources**: News articles, scientific literature, government reports, citizen science
- **Coverage**: Complex topography, active seismicity, and shifting precipitation patterns

### Data Structure
| Node Type | Count | Percentage | Description |
|-----------|-------|------------|-------------|
| Event | 1,558 | 61.1% | Core landslide event records |
| Source | 440 | 17.2% | Information sources and references |
| GazetteerPoint | 331 | 13.0% | Geographic reference points |
| LandslideProfile | 223 | 8.7% | Landslide characterization profiles |

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup Environment

```bash
# Create and activate conda environment
conda create --name ccgrmas python=3.8
conda activate ccgrmas

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Start the FastAPI server
uvicorn ccgrmas.app:app --reload
```

The application will be available at `http://localhost:8000`

## Performance Results

CC-GRMAS demonstrates significant improvements over traditional approaches:

| Approach | Method | F1-Score | Precision | Recall | Parameters |
|----------|--------|----------|-----------|---------|------------|
| Nepal Study | Random Forest | 0.56 | 0.47 | 0.70 | <0.1M |
| Nepal Study | XGBoost | 0.54 | 0.45 | 0.67 | <0.1M |
| Nepal Study | Gradient Boosting | 0.56 | 0.49 | 0.65 | <0.1M |
| Nepal Study | U-Net | 0.79 | 0.91 | 0.69 | 31.0M |
| **CC-GRMAS** | **Spatial GNN** | **0.7981** | **0.8062** | **0.7928** | **<0.1M** |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NASA Goddard Space Flight Center for the Global Landslide Catalog
- Global Precipitation Measurement (GPM) mission
- The communities of High Mountain Asia affected by landslide risks

---

**Note**: This project is submitted to the NeurIPS 2025 Tackling Climate Change with Machine Learning workshop. Repository details are anonymized for peer review.
