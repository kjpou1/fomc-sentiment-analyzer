# fomc-sentiment-analyzer

**fomc-sentiment-analyzer** is a Python-based tool designed for clause-level sentiment analysis of Federal Open Market Committee (FOMC) minutes and other financial texts. By leveraging the Sentiment Focus (SF) method and FinBERT, a financial domain-specific language model, this project aims to provide nuanced insights into the sentiments expressed within complex financial documents.

## Table of Contents

- [fomc-sentiment-analyzer](#fomc-sentiment-analyzer)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Background](#background)
  - [Technologies Used](#technologies-used)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Alternatives to SF](#alternatives-to-sf)
  - [Development Guide](#development-guide)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Features

- **Clause Segmentation**: Breaks down complex sentences into individual clauses for granular analysis.
- **FinBERT Integration**: Utilizes FinBERT for accurate sentiment classification tailored to the financial domain.
- **Sentiment Aggregation**: Combines clause-level sentiments to determine overall sentiment, with support for customizable weighting schemes.
- **Visualization**: Offers optional graphical representations of sentiment distributions using Matplotlib or Seaborn.
- **Extensibility**: Adaptable to various financial texts beyond FOMC minutes, such as earnings calls and financial reports.

## Background

Sentiment analysis is crucial in finance for interpreting market sentiment, assessing risk, and making informed investment decisions. Traditional sentence-level analysis may overlook the nuanced opinions expressed in complex sentences common in financial texts. By implementing the Sentiment Focus (SF) method, this tool performs clause-level sentiment analysis, providing a more detailed and accurate understanding of the sentiments conveyed.

## Technologies Used

- **Python 3.8+**
- **spaCy**: For natural language processing and clause segmentation.
- **NLTK**: For additional NLP tasks.
- **Transformers**: For integrating FinBERT into the sentiment analysis pipeline.
- **PyTorch**: As the backend framework for running FinBERT.
- **scikit-learn**: For model evaluation metrics.
- **Matplotlib/Seaborn**: (Optional) For data visualization.

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- **Virtual Environment**: Recommended to prevent dependency conflicts

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/fomc-sentiment-analyzer.git
   cd fomc-sentiment-analyzer
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate the virtual environment
   # On Windows:
   .venv\Scripts\activate

   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy Model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('punkt')
   ```

## Alternatives to SF

For a list of alternative approaches to the selected Sentiment Focus (SF) method as it applies to clause-level sentiment analysis, please refer to the [Alternative Approaches to the Sentiment Focus (SF) Method in Sentiment Analysis](ALTERNATIVES.md).


## Development Guide

For detailed instructions on setting up the development environment, acquiring data, and implementing the system, please refer to the [Development Guide](DEVELOPMENT.md).

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by the need for more nuanced sentiment analysis in financial texts.
- Utilizes the [FinBERT model](https://huggingface.co/yiyanghkust/finbert-tone) developed by Huawei Noah's Ark Lab.