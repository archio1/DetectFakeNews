import argparse
import pandas as pd
from FakeClass import DetectFake

def main():
    # open = subprocess.Popen(cmd, shell=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("way")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path_to = pd.read_csv(args.way)
    # print(f'args={args}\n\n , mode={args.mode}, args={args.create}')
    if args.mode == 'passive_aggressive' and args.create:
        model_pass_agr = DetectFake(model_name=args.mode, data=path_to)
        model_pass_agr.prepare_data()
        model_pass_agr.vectorization_of_text()
        model_pass_agr.passive_aggressive_classifier()
        model_pass_agr.train()
        model_pass_agr.save_model()
        print('module is done')
    elif args.mode == 'passive_aggressive' and args.check:
        model_pass_agr = DetectFake(model_name=args.mode, model_path='../resources/passive_aggressive_model')
        print(model_pass_agr.predict(path_to['text'][8]))
    elif args.mode == 'neural_network' and args.create:
        model_neural_network = DetectFake(model_name=args.mode, data=path_to)
        model_neural_network.cleaning_news_for_neural_net()
        model_neural_network.vector_text_for_neural_network()
        model_neural_network.create_matrix()
        model_neural_network.init_test_train_split()
        model_neural_network.neural_network()
        model_neural_network.save_model()
    elif args.mode == 'neural_network' and args.check:
        predict_model_neural_network = DetectFake(model_name=args.mode, model_path='../resources/neural_model',
                                                  newtext=path_to.iloc[7])
        predict_model_neural_network.predict_by_neural_network()

if __name__ == '__main__':
    main()

