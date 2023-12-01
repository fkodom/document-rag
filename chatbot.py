import os
import shutil

from document_rag.rag import RAG
from document_rag.settings import Settings

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "documents",
        type=str,
        nargs="+",
        help="One or more local paths to PDF documents.",
    )
    parser.add_argument(
        "--show-references",
        action="store_true",
        help="Show reference texts for each result.",
    )
    args = parser.parse_args()

    for path in args.documents:
        _, ext = os.path.splitext(path)
        if not os.path.exists(path):
            print(f"File '{path}' does not exist.")
            exit(1)
        if not ext.lower() == ".pdf":
            print(f"File extension '{ext}' for '{path}' not supported. Must be PDF.")
            exit(1)

    shutil.rmtree(Settings().DOCUMENT_RAG_VECTOR_DB_CACHE_DIR, ignore_errors=True)
    rag = RAG.from_settings()
    rag.add_pdf_documents(paths=args.documents, verbose=True)
    print("Ingested PDF documents. Please ask your questions.")

    while True:
        prompt = input(">>> ").replace("\n", "").strip()
        if prompt == "":
            continue
        elif prompt.lower() == "exit":
            break

        result = rag.generate(prompt=prompt)
        print(result["text"])
        if not args.show_references:
            continue

        print("\nReferences:")
        for reference in result["search_results"]:
            print()
            print(reference["metadata"]["path"], end="")
            start_page, end_page = reference["metadata"]["page_range"]
            if start_page == end_page:
                print(f"(p {start_page})")
            else:
                print(f"(pp {start_page}-{end_page})")
            print(reference["text"])
        print()
