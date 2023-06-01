import argparse
import os
import time
import pickle
import numpy as np
import src.utils_general as utils_general
import src.utils_eval as utils_eval
from src.configs import get_best_config
from src.diagnosis import hit_att, ndcg
from src.lat import LAT

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='dataset name', type=str, default='ASD',
                    choices=['MSL', 'SMAP', 'SMD', 'ASD'])
parser.add_argument("--entities", type=str,
                    # default='FULL',
                    default='omi-1',
                    help='FULL represents all the entities, or a list of entity names split by comma')
parser.add_argument('--device', help='torch device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--runs', help='', type=int, default='1')
parser.add_argument('--stride', help='', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--k', type=int, default=5)
args = parser.parse_args()

results_raw_dir = f'./&results/raw/{args.data}/'
results_raw_metrics_path = os.path.join(results_raw_dir, '@raw_results.csv')
os.makedirs(results_raw_dir, exist_ok=True)

results_avg_metrics_path = f'./&results/report/{args.data}-result.csv'
os.makedirs('./&results/report/', exist_ok=True)
print(f'results are recorded in {results_avg_metrics_path} and {results_raw_metrics_path}')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run(train_df, test_df, labels, name, model):

    train_df, test_df = utils_general.data_standardize(train_df, test_df)
    model.fit(train_df)
    prediction_dic = model.predict(test_df)

    data = name.split('-', 1)[0]
    if data == 'ASD' or data == 'SMD':
        channels = name.split('-', 1)[1]
        channel_labels = np.load(os.path.join(f'./datasets/{data}/Diagnoise/', f'{channels}_labels.npy'))

        Hit = hit_att(prediction_dic['error_tc'], channel_labels)
        NDCG = ndcg(prediction_dic['error_tc'], channel_labels)
    else:
        Hit = (0, 0)
        NDCG = (0, 0)

    prediction_dic = utils_general.meta_process_scores(prediction_dic, name)
    scores = prediction_dic['score_t']
    post_dic = model.predict(train_df)
    post_dic = utils_general.meta_process_scores(post_dic, name)
    post_s = post_dic['score_t']


    eval_info = utils_eval.get_metrics(labels, post_s, scores)
    adj_eval_info = utils_eval.get_metrics(labels, post_s, utils_eval.adjust_scores(labels, scores))
    return prediction_dic, eval_info, adj_eval_info + Hit + NDCG

def main():
    configs = get_best_config(args.data, args.num_epochs, args.seq_len, args.stride, args.k)
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f = open(results_raw_metrics_path, 'a')
    f2 = open(results_avg_metrics_path, 'a')
    print(f'Time: {cur_time}, Data: {args.data}, Runs: {args.runs} \n Configs: {configs} \n ', file=f)
    print(f'Time: {cur_time}, Data: {args.data}, Runs: {args.runs} \n Configs: {configs} \n', file=f2)
    print(f'data, adj_auroc, adj_aupr, adj_f1, adj_p, adj_r, Hit@100%, Hit@150%, NDCG@100%, NDCG@150%,'
          f'adj_auroc_std, adj_aupr_std, adj_f1_std, adj_p_std, adj_r_std, Hit@100%_std, Hit@150%_std, NDCG@100%_std, NDCG@150%_std, time', file=f2)
    f.close()
    f2.close()

    train_df_lst, test_df_lst, label_lst, name_lst = utils_general.get_data_lst(args.data, data_root,
                                                                                entities=args.entities)
    name_lst = [args.data + '-' + n for n in name_lst]

    for train, test, label, name in zip(train_df_lst, test_df_lst, label_lst, name_lst):
        entries = []
        t_lst = []
        for i in range(args.runs):
            t1 = time.time()
            print(f'\n\n Running {name} [{i+1}/{args.runs}]')
            configs['seed'] = 42 + i

            model = LAT(**configs)
            predictions, eval_metrics, adj_eval_metrics = run(train, test, label, name, model)
            entries.append(adj_eval_metrics)
            t = time.time() - t1
            t_lst.append(t)
            # save prediction raw results
            prediction_path = os.path.join(results_raw_dir, name + '-'+ str(i) + '.pkl')
            f = open(prediction_path, 'wb')
            pickle.dump(predictions, f)
            f.close()

            # save evaluation metrics raw results
            txt = f'{name},'
            txt += ', '.join(['%.4f' % a for a in eval_metrics]) + ', pa, ' + ', '.join(['%.4f' % a for a in adj_eval_metrics])
            txt += f', runs, {i+1}/{args.runs}'

            f = open(results_raw_metrics_path, 'a')
            print(txt)
            print(txt, file=f)
            f.close()

        avg_entry = np.average(np.array(entries), axis=0)
        std_entry = np.std(np.array(entries), axis=0)
        avg_t = np.average(np.array(t_lst))

        f2 = open(results_avg_metrics_path, 'a')
        txt = f'{name}, ' + ", ".join(['%.4f' % a for a in np.hstack([avg_entry, std_entry])]) + f', {avg_t:.1f} s'
        print(txt)
        print(txt, file=f2)

if __name__ == '__main__':
    data_root = f'./datasets/'
    main()
