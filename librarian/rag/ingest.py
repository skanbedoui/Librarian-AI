"""
STEP 1 (offline): Load the PDFs and cut them into chunks.

Each chunk remembers where it came from (metadata):
    doc title, program, PDF page number, and the section heading it sits under.
That metadata is what lets us show citations like  [Engineering Guide, p.21, "Grading Scale"].
"""
import re
from pypdf import PdfReader

from .config import DOCS_DIR, DOCUMENTS, CHUNK_WORDS, CHUNK_OVERLAP

# "VII. Assessment and Grades"  -> top-level section (roman numeral)
SECTION_RE = re.compile(r"^([IVX]+)\.\s+([A-Z][^.]{2,80})$")
# "5. Grading Scale"            -> sub-section (number)
SUBSECTION_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^.]{2,80})$")


def clean_line(line: str) -> str:
    line = re.sub(r"(\.\s?){4,}", " ", line)   # remove "......" leaders from the table of contents
    line = re.sub(r"\s+", " ", line)           # collapse runs of spaces
    return line.strip()


def is_table_of_contents(lines: list[str]) -> bool:
    """TOC pages are full of dotted leaders; they only add noise to search."""
    dotted = sum(1 for l in lines if "...." in l or ". . . ." in l)
    return dotted > 5


def load_pages():
    """Yield one dict per PDF page, with clean text and the section it belongs to."""
    for filename, (title, program) in DOCUMENTS.items():
        reader = PdfReader(DOCS_DIR / filename)
        section, subsection = "", ""
        for page_no, page in enumerate(reader.pages, start=1):
            raw_lines = (page.extract_text() or "").splitlines()
            if is_table_of_contents(raw_lines):
                continue
            lines = [clean_line(l) for l in raw_lines]
            lines = [l for l in lines if l and not l.isdigit()]   # drop empty lines and page numbers

            # Walk the lines, tracking the current heading. We tag each line with its heading
            # so a chunk that starts mid-page still knows which section it is in.
            tagged = []
            for line in lines:
                if m := SECTION_RE.match(line):
                    section, subsection = line, ""
                elif m := SUBSECTION_RE.match(line):
                    subsection = line
                heading = " > ".join(h for h in (section, subsection) if h)
                tagged.append((line, heading))

            if tagged:
                yield {"doc": title, "file": filename, "program": program,
                       "page": page_no, "lines": tagged}


def chunk_pages(pages, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """
    Sliding window over the words of each page.

    Why per page? Citations point to a page, and the guides are short enough per page
    that one page is usually 1-3 chunks. Why overlap? So a sentence cut at a boundary
    still appears whole in one of the two neighbouring chunks.
    """
    chunks = []
    for p in pages:
        # Turn the page into a flat list of (word, heading) pairs
        words = [(w, heading) for line, heading in p["lines"] for w in line.split()]
        step = max(1, chunk_words - overlap)
        for start in range(0, len(words), step):
            window = words[start:start + chunk_words]
            if len(window) < 15 and start > 0:   # tiny leftover tail already covered by overlap
                break
            chunks.append({
                "id": f"{p['program']}-p{p['page']}-{start // step}",
                "doc": p["doc"],
                "file": p["file"],
                "program": p["program"],
                "page": p["page"],
                "section": window[0][1],          # heading at the start of the chunk
                "text": " ".join(w for w, _ in window),
            })
            if start + chunk_words >= len(words):
                break
    return chunks
