import argparse

from .config import load_settings
from .engine import create_engine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    engine = create_engine(load_settings())
    prepared = engine.prepare(args.question)
    for token in engine.answer(args.question,prepared):
        print(token,end="",flush=True)
    print("\n证据：",prepared.evidence)

if __name__=="__main__":
    main()