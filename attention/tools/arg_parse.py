import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--language_model', default='transformer')

    parser.add_argument('--data_pkl', default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_word_vec', type=int, default=512)
    parser.add_argument('--d_inner_hid', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')
    parser.add_argument('--scale_emb_or_prj', type=str, default='prj')

    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--use_tb', action='store_true')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--label_smoothing', action='store_true')

    # transformer preprocess related
    spacy_support_langs = ['de', 'el', 'en', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']
    parser.add_argument('--lang_src', required=True, choices=spacy_support_langs)
    parser.add_argument('--lang_trg', required=True, choices=spacy_support_langs)
    parser.add_argument('--save_data', required=True)
    parser.add_argument('--data_src', type=str, default=None)
    parser.add_argument('--data_trg', type=str, default=None)

    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--min_word_count', type=int, default=3)
    parser.add_argument('--keep_case', action='store_true')
    parser.add_argument('--share_vocab', action='store_true')

    # bert preprocess related
    parser.add_argument("--movie_conversations", required=False)
    parser.add_argument("--movie_lines", required=False)
    parser.add_argument("--raw_text", required=False)
    parser.add_argument("--output", required=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda

    return args