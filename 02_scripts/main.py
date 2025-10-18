# ilm_project/02_scripts/main.py
import argparse, time, sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_01_extract import run_extraction
from step_02_chunk import run_chunking
from step_03_consolidate import run_consolidation

def main():
    parser = argparse.ArgumentParser(description="ILM Data Processing Pipeline")
    parser.add_argument("step", choices=["all","extract","chunk","consolidate","generate_qa"])
    args = parser.parse_args()

    start = time.time()
    print(f"--- Starting Pipeline: Step '{args.step}' ---")

    if args.step == "all":
        print("\n[1/4] Extract"); run_extraction()
        print("\n[2/4] Chunk"); run_chunking()
        print("\n[3/4] Consolidate"); run_consolidation()
        print("\n[4/4] Generate Q&A")
        from step_04_generate_qa import run_qa_generation
        run_qa_generation()
    elif args.step == "extract":
        run_extraction()
    elif args.step == "chunk":
        run_chunking()
    elif args.step == "consolidate":
        run_consolidation()
    elif args.step == "generate_qa":
        from step_04_generate_qa import run_qa_generation
        run_qa_generation()

    print(f"\n--- Pipeline finished in {time.time()-start:.2f} seconds ---")

if __name__ == "__main__":
    main()
