"""Command-line Q&A.   Usage:  python ask.py "How do taints work?"   (or run with no args for a loop)"""
import sys

from rag import answer


def show(question: str):
    result = answer(question)
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for i, c in enumerate(result["chunks"], 1):
        print(f"  [{i}] {c['source']}  ({c['title']}, distance {c['distance']:.3f})")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        while True:
            q = input("Question (blank to quit): ").strip()
            if not q:
                break
            show(q)
