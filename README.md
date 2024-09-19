# fomc-sentiment-analyzer

**fomc-sentiment-analyzer** is a Python-based tool designed for clause-level sentiment analysis of Federal Open Market Committee (FOMC) minutes and other financial texts. By leveraging the Sentiment Focus (SF) method and FinBERT, a financial domain-specific language model, this project aims to provide nuanced insights into the sentiments expressed within complex financial documents.

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

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Inspired by the need for more nuanced sentiment analysis in financial texts.
- Utilizes the [FinBERT model](https://huggingface.co/yiyanghkust/finbert-tone) developed by Huawei Noah's Ark Lab.
