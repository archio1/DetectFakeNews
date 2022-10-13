import argparse
import pandas as pd
from FakeClass import DetectFake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("way")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path_to = pd.read_csv(args.way)
    if args.mode == "passive_aggressive" and args.create:
        model_pass_agr = DetectFake(model_name=args.mode, data=path_to)
        model_pass_agr.operations_of_create_PAM()
        print("module is done")
    elif args.mode == "passive_aggressive" and args.check:
        model_pass_agr = DetectFake(
            model_name=args.mode, model_path="../resources/passive_aggressive_model"
        )
        print(model_pass_agr.predict(path_to["text"][8]))
    elif args.mode == "neural_network" and args.create:
        model_neural_network = DetectFake(model_name=args.mode, data=path_to)
        model_neural_network.operations_of_create_NNM()
    elif args.mode == "neural_network" and args.check:
        predict_model_neural_network = DetectFake(
            model_name=args.mode,
            data=path_to.iloc[7],
            model_path="../resources/neural_model",
        )
        predict_model_neural_network.predict_by_neural_network()


if __name__ == "__main__":
    main()
