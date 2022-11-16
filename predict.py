import argparse
import json
import os, time, logging
from data_preprocessing import load_data
from rank_mlp import ranking
from interact_embeddings import save_interact_embeddings


parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--input', type=str, default='../phase_2/test/', help='input path of the dataset directory.')
parser.add_argument('--output', type=str, default='../phase_2/', help='output path of the prediction file.')
parser.add_argument('--encoder_path', type=str, default='saved/bsz2_lr1e-05/lawformer.pt', help='encoder path.')
parser.add_argument('--rank_mlp_path', type=str, default='saved/bsz1_lr5e-05/rank_mlp.pt', help='rank_mlp path.')

args = parser.parse_args()
input_path = args.input
input_query_path = os.path.join(input_path, 'query.json')
input_candidate_path = os.path.join(input_path, 'candidates')
output_path = args.output
new_test_data_path = os.path.join(os.path.dirname(__file__), 'new_test_data.json')
new_test_embedding_path = os.path.join(os.path.dirname(__file__), 'new_test_embeddings')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    print('begin...')
    if not os.path.exists(new_data_path):
        load_data(query_file=input_query_path,
                  candidate_dir=input_candidate_path,
                  saved_test_file=new_test_data_path,
                  data_type='test')
    time.sleep(1)
    print('new data converting finished...')

    if not os.path.exists(new_data_path):
        save_interact_embeddings(saved_data=new_test_data_path,
                                 saved_embeddings=new_test_embedding_path,
                                 saved_encoder_path=args.encoder_path)
    print('interact embeddings saved...')

    print('prediction starting...')
    result = ranking(mode='test',
                     test_data=new_test_data_path,
                     test_embeddings=new_test_embedding_path,
                     rank_mlp_path=args.rank_mlp_path)
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    main()
