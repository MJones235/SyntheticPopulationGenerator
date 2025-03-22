import argparse
from src.evaluation.sampler import Sampler

def main():
    parser = argparse.ArgumentParser(description="Stratified sample generator for LLM variable input data.")
    parser.add_argument("--variable", type=str, required=True, help="The name of the variable to sample (e.g. population_size, age_distribution)")

    args = parser.parse_args()
    variable = args.variable

    sampler = Sampler(variable)
    sampler.sample()

if __name__ == "__main__":
    main()