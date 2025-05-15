# README

This project provides an interactive pipeline for generating requirements from a PDF, reviewing them, automatically
generating implementation details, iteratively refining code, and finally generating documentation. The application is
built with Streamlit, incorporates several agents for processing, and offers a live preview via an embedded HTTP server.

The documentation is divided into multiple topics:

- [Installation and Setup](Installation.md)
- [Running the Pipeline](Running-the-Pipeline.md)


## Known Bugs

- Do the iteration of coding agent till the maximum time allowed (i.e. 3) else, the Documentation Agent will not work as
  expected. We are trying to fix it ASAP.
- We are not storing the Documentation Agent in a file but, just showing in the UI for demo purposes.