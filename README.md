# document-rag

A simple retrieval-augmented generation (RAG) system for answering questions about PDF documents.  

> **Note:** RAG responses do not depend on chat history -- only the provided documents.  Each question should be self-contained, and not depend on previous questions or answers.


## Getting Started

Clone the repo and install with `pip`:

```bash
# clone
gh repo clone fkodom/document-rag
cd document-rag

# install
pip install -e .

# for contributors only:
# - install test dependencies
# - setup pre-commit hooks
pip install -e ".[test]"
pre-commit install
```

Run the `chatbot.py` script:

```bash
python chatbot.py \
    path/to/document-1.pdf \
    path/to/document-2.pdf \
    ...
```

This will start a simple Q&A session.  Sample PDFs are provided in the `assets/` folder.  For example:

```bash
python chatbot.py ./assets/alice-in-wonderland.pdf
```
```bash
Extracting PDFs: 100%|██████████| 1/1 [00:01<00:00,  1.63s/it]
Ingested PDF documents. Please ask your questions.
>>> What is Alice's cat's name?
Dinah
>>> What characters are present at the tea party?
The March Hare, the Hatter, and the Dormouse are present at the tea party.
```

Responses are not fully deterministic, so you may get slightly different answers each time.

Specify the `--show-references` flag to see which documents/pages were used to answer each question.  By default, 5 documents are used.

```bash
Extracting PDFs: 100%|██████████| 1/1 [00:01<00:00,  1.63s/it]
Ingested PDF documents. Please ask your questions.
>>> What is Alice's cat's name?
Dinah

References:

...

.../assets/alice-in-wonderland.pdf (pp 13-14)
passionate voice. ‘Would YOU like cats if you were me?’ ‘Well, perhaps not,’ said Alice in a soothing tone: ‘don’t be angry about it. And yet I wish I could show you our cat Dinah: I think you’d take a fancy to cats if you could only see her. She is such a dear quiet thing,’ Alice went on, half to herself, as she swam lazily about in the pool, ‘and she sits 23 purring so nicely by the ﬁre, licking her paws and washing her face–and she is such a nice soft thing to nurse–and she’s such a capital one for catching mice–oh, I beg your pardon!’ cried Alice again, for this time the Mouse was bristling all over, and she felt certain it must be really

.../assets/alice-in-wonderland.pdf (p 38)
right way to change them–’ when she was a little startled by seeing the Cheshire Cat sitting on a bough of a tree a few yards o↵. The Cat only grinned when it saw Alice. It looked good-natured, she thought: still it had VERY long claws and a great many teeth, so she felt that it ought to be treated with respect. ‘Cheshire Puss,’ she began, rather timidly, as she did not at all know whether it would like the name: however, it only grinned a little wider. ‘Come, it’s pleased so far,’ thought Alice, and she went on. ‘Would you tell me, please, which way I ought to go from here?’ ‘That depends a good deal on where you want to get to,’ said the Cat. ‘I
```


## Tests and Linting

| Tool | Description | Runs on |
| --- | --- | --- |
| [black](https://github.com/psf/black) | Code formatter | - `git commit` (through `pre-commit`) <br> - `git push` <br> - pull requests |
| [ruff](https://github.com/astral-sh/ruff) | Code linter | - `git commit` (through `pre-commit`) <br> - `git push` <br> - pull requests |
| [pytest](https://github.com/pytest-dev/pytest) | Unit testing framework | - `git push` <br> - pull requests |
| [mypy](https://github.com/python/mypy) | Static type checker | - `git push` <br> - pull requests |
| [pre-commit](https://github.com/pre-commit/pre-commit) | Pre-commit hooks | - `git commit` |
