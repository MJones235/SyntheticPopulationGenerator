import argparse
from src.evaluation.sampler import Sampler

def main():
    parser = argparse.ArgumentParser(description="Stratified sample generator for LLM variable input data.")
    parser.add_argument("--variable", type=str, required=True, help="The name of the variable to sample (e.g. population_size, age_distribution)")
    parser.add_argument("--mode", type=str, choices=["sample", "distribution"], default="sample", help="Sampling mode: 'sample' for stratified raw data, 'distribution' for age/sex percentage pivot")

    args = parser.parse_args()
    variable = args.variable
    mode = args.mode

    sampler = Sampler(variable)
    sampler.sample(mode=mode)

if __name__ == "__main__":
    main()