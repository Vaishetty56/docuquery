# Dataset

**Topic: trustworthy and responsible AI (US NIST technical publications).**
One focused topic, 6 documents, all from one publisher.

## Licensing basis
NIST states that works authored by NIST employees are not subject to copyright protection in the
United States (17 U.S.C. section 105), and that foreign rights are reserved. Source of the statement:
https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications
This is a public-domain-in-the-US situation, **not a Creative Commons licence**. For a local portfolio
project that is fine, but:
- Check the cover/front matter of EACH PDF for non-NIST (contractor) authorship or a copyright notice.
  NIST may hold copyright in works by non-employees when it is assigned to NIST.
- Do not commit the PDFs to a public repo unless you have checked each one. Linking to the URLs and
  letting users download them themselves is always safe. `data/pdfs/` is where you put them locally.

## Documents
Status key: **URL confirmed** = I found this exact URL in search results on 2026-10-04.
**You must confirm** = identified by NIST publication number only; open the NIST publication page
(search the number on nist.gov or doi.org/10.6028/NIST.<ID>) and download from there.

| # | Document | Source | Status |
|---|---|---|---|
| 1 | NIST AI 100-1, Artificial Intelligence Risk Management Framework (AI RMF 1.0), Jan 2023 | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | URL confirmed |
| 2 | NIST AI 600-1, AI RMF Generative AI Profile | NIST publication page for AI 600-1 | You must confirm |
| 3 | NIST AI 100-2, Adversarial Machine Learning: taxonomy and terminology (latest edition) | NIST publication page for AI 100-2 | You must confirm |
| 4 | NIST SP 1270, Towards a Standard for Identifying and Managing Bias in AI | NIST publication page for SP 1270 | You must confirm |
| 5 | NISTIR 8312, Four Principles of Explainable AI | NIST publication page for IR 8312 | You must confirm |
| 6 | NIST AI 100-4, Reducing Risks Posed by Synthetic Content | NIST publication page for AI 100-4 | You must confirm |

Replace any document you cannot confirm with another NIST AI publication on the same theme.
Fill in the table below once you have downloaded them (the eval uses the file names as citations, so
keep file names exactly as saved).

## Your record (fill in)
| File name in data/pdfs/ | Exact source URL | Date downloaded | Copyright notice on cover? | Pages |
|---|---|---|---|---|
| | | | | |

## Known dataset limitations
- One publisher and one writing style: results may not carry over to other document types
  (tables-heavy reports, scanned PDFs, non-English text).
- Several documents cite each other and share terminology, which makes some questions hard to
  answer from a single document: good for stress-testing, but check labels carefully.
- Tables and figures are extracted as plain text by pypdf and may be garbled.
