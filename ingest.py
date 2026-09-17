"""
Manual textbook indexing.

    python ingest.py book.pdf          index one PDF
    python ingest.py --all             index everything in books/
    python ingest.py --list            show what is indexed
    python ingest.py --search "query"  test retrieval

Normally you do not need this: web.py indexes books/ automatically at startup.
"""
import sys

import config
import rag


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    rag.library.load_from_disk()

    if argv[0] == "--list":
        rows = rag.library.summary()
        if not rows:
            print("No textbooks indexed. Put PDFs in books/ and run: python ingest.py --all")
        for row in rows:
            print(f"  {row['book']:<40} {row['chunks']} chunks")
        return 0

    if argv[0] == "--all":
        rag.library.index_books_folder()
        return 0

    if argv[0] == "--search":
        if len(argv) < 2:
            print("usage: python ingest.py --search \"your question\"")
            return 1
        hits = rag.library.search(" ".join(argv[1:]))
        if not hits:
            print("No matches above the similarity threshold "
                  f"({config.RAG_MIN_SCORE}).")
        for h in hits:
            print(f"\n--- {h['book']} p.{h['page']}  (score {h['score']:.3f})")
            print(h["text"][:400].replace("\n", " "))
        return 0

    for path in argv:
        rag.library.index_pdf(path, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
