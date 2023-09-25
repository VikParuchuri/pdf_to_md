# Convert PDFs to markdown

- Extract text from pdf with pymupdf
- Remove headers/footers using clustering with DBScan algorithm
- Convert text to markdown with a finetuned LLM

Known issues: it will repeat text if the generation goes off the rails.  I need to retrain the model using some lessons from the nougat paper.

## Installation

- `poetry install`

## Usage

