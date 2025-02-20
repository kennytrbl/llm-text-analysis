# llm-text-analysis

A website for extracting, summarizing, and querying PDFs with AI.

The goal of this project is to create an AI-powered document processing tool that extracts, summarizes, and answers questions based on PDF content using large language models. It is designed for professionals, researchers, and students who need to quickly analyze and extract key insights from large documents.

## Installation

Install llm-text-analysis with pip

```bash
  pip install pdfplumber
  pip install --upgrade transformers
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install ipywidgets
  pip install nltk
  pip install tiktoken
  pip install protobuf
  pip install blobfile
```

## Usage/Examples

```python
import pdfplumber

pdf_path = "./content/alan_turing_ wikipedia.pdf" # edit

output_text_file = "extracted_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()

with open(output_text_file, "w", encoding="utf-8") as text_file:
    text_file.write(extracted_text)

print(f"Text extracted and saved to {output_text_file}")
```

    Text extracted and saved to extracted_text.txt

```python
with open("./extracted_text.txt", "r") as file:
    document_text = file.read()

print(document_text[:500])
```

    2/20/25, 10:46 AM Alan Turing - Wikipedia
    Alan Turing
    Alan Mathison Turing (/ËˆtjÊŠÉ™rÉªÅ‹/; 23 June 1912 â€“ 7
    Alan Turing
    June 1954) was an English mathematician, computer
    OBE FRS
    scientist, logician, cryptanalyst, philosopher and
    theoretical biologist.[5] He was highly influential in the
    development of theoretical computer science, providing
    a formalisation of the concepts of algorithm and
    computation with the Turing machine, which can be
    considered a model of a general-purpose
    computer.[6][7

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summary = summarizer(document_text[:2000], max_length=512, min_length=100, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

    Device set to use cuda:0
    Your max_length is set to 512, but your input_length is only 506. Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer('...', max_length=253)


    Summary: Alan Turing (23 June 1912 - 7 June 1954) was an English mathematician, computer, logician, cryptanalyst, philosopher and theoretical biologist. During World War II, Turing worked for the Government Code and Cypher School at Bletchley Park, Britain's codebreaking centre. He led Hut 8, the section responsible for Turing in 1951 for German naval cryptanalysis. Turing devised techniques for speeding the breaking of German ciphers, including 23 June 1912improvements to the pre-war Polish bomba method.

```python
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(document_text)

passages = []
current_passage = ""
for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < 400:  # adjust word limit as needed
        current_passage += " " + sentence
    else:
        passages.append(current_passage.strip())
        current_passage = sentence
if current_passage:
    passages.append(current_passage.strip())
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     C:\Users\Kenny\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!

```python
qg_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_questions_pipeline(passage, min_questions=3):
    input_text = f"Generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('\n')

    questions = [q.strip() for q in questions if q.strip()]

    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"Generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('\n')
            questions.extend([q.strip() for q in additional_questions if q.strip()])
    return questions[:min_questions]

for idx, passage in enumerate(passages):
    questions = generate_questions_pipeline(passage)
    print(f"Passage {idx+1}:\n{passage}\n")
    print("Generated Questions:")
    for q in questions:
        print(f"- {q}")
    print(f"\n{'-'*50}\n")

```

    Device set to use cuda:0
    Token indices sequence length is longer than the specified maximum sequence length for this model (642 > 512). Running this sequence through the model will result in indexing errors


    Passage 1:
    2/20/25, 10:46 AM Alan Turing - Wikipedia
    Alan Turing
    Alan Mathison Turing (/ËˆtjÊŠÉ™rÉªÅ‹/; 23 June 1912 â€“ 7
    Alan Turing
    June 1954) was an English mathematician, computer
    OBE FRS
    scientist, logician, cryptanalyst, philosopher and
    theoretical biologist. [5] He was highly influential in the
    development of theoretical computer science, providing
    a formalisation of the concepts of algorithm and
    computation with the Turing machine, which can be
    considered a model of a general-purpose
    computer. [6][7][8] Turing is widely considered to be the
    father of theoretical computer science. [9]
    Born in London, Turing was raised in southern
    England. He graduated from King's College, Cambridge,
    and in 1938, earned a doctorate degree from Princeton
    University. During World War II, Turing worked for the
    Government Code and Cypher School at Bletchley Park,
    Britain's codebreaking centre that produced Ultra
    intelligence. He led Hut 8, the section responsible for Turing in 1951
    German naval cryptanalysis. Turing devised techniques Born Alan Mathison Turing
    for speeding the breaking of German ciphers, including 23 June 1912
    improvements to the pre-war Polish bomba method, an Maida Vale, London, England
    electromechanical machine that could find settings for
    Died 7 June 1954 (aged 41)
    the Enigma machine. He played a crucial role in
    Wilmslow, Cheshire, England
    cracking intercepted messages that enabled the Allies to
    Cause of Cyanide poisoning as an act of
    defeat the Axis powers in many engagements, including
    death suicide[note 1]
    the Battle of the Atlantic. [10][11]
    Alma mater University of Cambridge (BA, MA)
    After the war, Turing worked at the National Physical
    Princeton University (PhD)
    Laboratory, where he designed the Automatic
    Known for Cryptanalysis of the Enigma Â·
    Computing Engine, one of the first designs for a stored-
    Turing's proof Â· Turing machine Â·
    program computer. In 1948, Turing joined Max
    Turing test Â· unorganised
    Newman's Computing Machine Laboratory at the
    machine Â· Turing pattern Â· Turing
    Victoria University of Manchester, where he helped
    reduction Â· "The Chemical Basis
    develop the Manchester computers[12] and became
    of Morphogenesis" Â· Turing
    interested in mathematical biology. Turing wrote on the
    paradox
    chemical basis of morphogenesis[13][1] and predicted
    Awards Smith's Prize (1936)
    oscillating chemical reactions such as the Belousovâ€“
    Scientific career
    https://en.wikipedia.org/wiki/Alan_Turing 1/382/20/25, 10:46 AM Alan Turing - Wikipedia
    Zhabotinsky reaction, first observed in the 1960s.

    Generated Questions:
    - What was the name of the chemical reaction that Turing predicted?
    - What was Alan Turing's profession?
    - What was the name of the computer that Turing developed?

```python
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_unique_questions(passages, qa_pipeline):
    answered_questions = set()

    for idx, passage in enumerate(passages):
        questions = generate_questions_pipeline(passage)

        for question in questions:
            if question not in answered_questions:
                answer = qa_pipeline({'question': question, 'context': passage})
                print(f"Q: {question}")
                print(f"A: {answer['answer']}\n")
                answered_questions.add(question)
        print(f"{'='*50}\n")

answer_unique_questions(passages, qa_pipeline)
```

    Device set to use cuda:0
    C:\Users\Kenny\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\pipelines\question_answering.py:391: FutureWarning: Passing a list of SQuAD examples to the pipeline is deprecated and will be removed in v5. Inputs should be passed using the `question` and `context` keyword arguments instead.
      warnings.warn(


    Q: What was the name of the chemical reaction that Turing predicted?
    A: Belousovâ

    Q: What was Alan Turing's profession?
    A: mathematician

    Q: What was the name of the computer that Turing developed?
    A: Manchester

    ==================================================

## Features

- **PDF Text Extraction**: Uses `pdfplumber` to extract text from PDF files and save it as a .txt file.
- **Text Summarization**: Utilizes the `facebook/bart-large-cnn` model to generate concise summaries of extracted text.
- **Text Tokenization**: Splits extracted text into sentences using `nltk` for better processing.
- **Passage Segmentation**: Groups sentences into passages of a defined word limit for structured analysis.
- **Automated Question Generation**: Uses `google/flan-t5-base` to generate questions based on segmented passages.
- **Question Answering**: Implements `deepset/roberta-base-squad2` to generate answers to the auto-generated questions from the text.
- **GPU Acceleration**: Supports CUDA for faster model inference where available.

## Tech Stack

Jupyter Notebook, Python, pdfplumber, nltk, Hugging Face Transformers, facebook/bart-large-cnn, google/flan-t5-base, deepset/roberta-base-squad2, PyTorch

## License

[MIT](https://choosealicense.com/licenses/mit/)
