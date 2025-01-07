# Timeline Long Meeting Summarization (TLMS) Research Project

A novel approach combining Knowledge Graphs and Large Language Models for automated timeline-based meeting summarization.

## 🎯 Research Overview

This project investigates Timeline Long Meeting Summarization (TLMS) - an automated approach to create time-based summaries of meetings that capture key events and their progression. We present a unified approach combining Knowledge Graphs (KG) with Large Language Models (LLMs) to tackle this challenging task.

## 📊 Key Features

- Knowledge Graph construction from meeting transcripts
- Integration of multiple LLM architectures
- Timeline-driven summarization
- Comprehensive evaluation metrics
- State-of-the-art performance analysis

## 🛠️ Technical Stack

### Language Models Evaluated
- FLAN-T5 (780M params)
- Falcon (7B params)
- GPT-3.5 (20B params)
- Long-LLaMA (3B params)
- LLaMA2 (7B params)
- LLaMA3 (8B params)

### System Requirements
- Ubuntu 18.04.5 LTS
- NVIDIA GeForce RTX 3090 (24 GB)
- RAM: 220GB
- Python 3.10.11

## 📋 Dataset

We utilize the MeetingBank dataset, which includes:
- 1,366 meetings
- Data from six major US cities
- Average meeting duration: 2.6 hours
- Average transcript length: 28,000+ tokens

### Dataset Statistics
| City | Meetings | Hours | Tokens | Speakers | Period |
|------|----------|--------|---------|-----------|---------|
| Denver | 401 | 979 | 25,460 | [3,20] | 2014-22 |
| Seattle | 327 | 446 | 15,045 | [3,14] | 2015-22 |
| Long Beach | 310 | 1103 | 39,618 | [4,19] | 2014-22 |
| Alameda | 164 | 730 | 47,981 | [2,15] | 2015-22 |
| King County | 132 | 247 | 20,552 | [2,10] | 2016-22 |
| Boston | 32 | 72 | 23,291 | [4,11] | 2021-22 |

## 🚀 Methodology

1. **Knowledge Graph Construction**
   - Utilize REBEL Large Model for relationship extraction
   - Transform unstructured data into triplets
   - Create time-annotated quadruples [Head, Relation, Tail, Minute]

2. **LLM Integration**
   - Apply quadruples linearly to LLMs
   - Generate timeline-based summaries
   - Maintain temporal coherence

3. **Evaluation Metrics**
   - Concat F1: Evaluates overall summary quality
   - Agree F1: Assesses minute-by-minute accuracy
   - ROUGE-1 and ROUGE-2 scores for both metrics

## 📈 Performance Metrics

### Implementation Details
```python
hyperparameters = {
    'temperature': 0.5,
    'top_p': 1,
    'top_k': 50
}
```

### Key Results
| Model | Concat F1 (R1/R2) | Agree F1 (R1/R2) |
|-------|-------------------|------------------|
| GPT-3.5 | 0.123/0.039 | 0.122/0.036 |
| LLaMA3 | 0.091/0.028 | 0.085/0.022 |
| FLAN-T5 | 0.045/0.012 | 0.058/0.013 |
| Falcon | 0.032/0.003 | 0.029/0.012 |
| LLaMA2 | 0.026/0.006 | 0.031/0.006 |
| Long-LLaMA | 0.016/0.005 | 0.020/0.006 |

## 💻 Installation & Usage

1. Clone the repository
```bash
git clone [repository-url]
cd timeline-meeting-summarization
```

2. Set up environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure model parameters
```bash
# Edit config.yaml with appropriate model settings and paths
```

4. Run experiments
```bash
python run_experiments.py --model [model_name] --dataset [dataset_path]
```

## 📚 Citations

If you use this code or findings in your research, please cite:
```bibtex
[Citation to be added after publication at CIKM'24]
```

## 🤝 Contributing

Contributions to improve the research implementation are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

For questions or collaboration opportunities, please reach out to the research team.

## 🙏 Acknowledgments

- MeetingBank dataset contributors
- REBEL Large Model team
- Computing infrastructure support

## 📊 Future Work

- Exploration of Few-shot prompting
- Investigation of Chain of Thoughts (CoT) prompting
- Implementation of Dynamic prompting
- Analysis of AutoCoT approaches
