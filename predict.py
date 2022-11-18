import argparse
import json
import torch
import os, time, logging
from data_preprocessing import load_data
from rank_mlp import ranking, RankMLP
from interact_extract import save_interact_embeddings
from transformers import AutoTokenizer
from utils import seed_everything


parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('-device', default='cuda', type=str)
parser.add_argument('-input', type=str, default='../data/cail2022_类案检索_封测阶段/', help='input path of the dataset directory.')
parser.add_argument('-output', type=str, default='./', help='output path of the prediction file.')
parser.add_argument('-encoder_path', type=str, default='saved/bsz1_lr1e-05/lawformer_best.pt', help='encoder path.')
parser.add_argument('-rank_mlp_path', type=str, default='./saved/bsz1_lr5e-05/rank_mlp_best.pt', help='rank_mlp path.')
parser.add_argument('-model_path', default='../../pretrained_model/lawformer', type=str)
parser.add_argument('-tokenizer_path', default="../../pretrained_model/roberta", type=str)
parser.add_argument('-extract_batch_size', default=10, type=int)
parser.add_argument('-max_length', default=1533, type=int)
parser.add_argument('-seed', default=42, type=int)
args = parser.parse_args()
seed_everything(args.seed)
input_path = args.input
input_query_path = os.path.join(input_path, 'query_cail22_stage3.json')
input_candidate_path = os.path.join(input_path, 'candidates_cail22_stage3')
output_path = args.output
new_test_data_path = os.path.join(os.path.dirname(__file__), 'saved/final_test_data.json')
new_test_embedding_path = os.path.join(os.path.dirname(__file__), 'saved/final_test_embeddings.npy')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%y-%m-%d %H:%M:%S')
    print('begin...')
    if not os.path.exists(new_test_data_path):
        load_data(query_file=input_query_path,
                  candidate_dir=input_candidate_path,
                  saved_test_file=new_test_data_path,
                  data_type='test')
    time.sleep(1)
    print('new data converting finished...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_tokens(['☢'])
    if not os.path.exists(new_test_embedding_path):
        save_interact_embeddings(args=args,
                                 tokenizer=tokenizer,
                                 saved_data=new_test_data_path,
                                 saved_embeddings=new_test_embedding_path,
                                 saved_encoder_path=args.encoder_path)
    print('interact embeddings saved...')

    print('prediction starting...')
    model = torch.load(args.rank_mlp_path)
    result = ranking(args=args,
                     mode='test',
                     model=model,
                     test_data=new_test_data_path,
                     test_embeddings=new_test_embedding_path)
    json.dump(result, open(os.path.join(output_path, 'prediction.json'), "w", encoding="utf8"), indent=2,
              ensure_ascii=False, sort_keys=True)
    print('output done.')


if __name__ == "__main__":
    main()
