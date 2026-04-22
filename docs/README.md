# Document Corpus

This directory contains authoritative and curated documents used for
demonstrating a Retrieval-Augmented Generation (RAG) assistant.

## Structure

- `raw/`
  Original source documents (PDFs) preserved for traceability.
- `curated/`
  Cleaned, scoped, and optimized markdown documents used for retrieval.

## Curated Documents

The `curated/` directory contains the following documents:

- `nist_800_53_selected_controls.md`: Selected security controls from NIST SP 800-53 Rev. 5
- `nist_800_61_incident_response.md`: Incident handling guidance from NIST SP 800-61 Rev. 2

Each document is split into sections during ingestion, with each section
becoming an individual retrieval unit for better granularity.

## Sources

All documents originate from publicly available U.S. government
publications (NIST Special Publications) that are in the public domain.

Curated documents intentionally summarize and scope content to improve
retrieval quality and explainability. They focus on enterprise security
and compliance topics relevant to the RAG assistant demonstration.

## Usage

Documents in `curated/` are automatically ingested when the `/ingest`
endpoint is called. The system uses semantic search to find relevant
sections based on user questions.