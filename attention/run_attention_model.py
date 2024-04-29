import random
import numpy as np
import os
import importlib
import torch
import sys
sys.path.append(os.getcwd())

from attention.tools.arg_parse import parse_args


if __name__ == "__main__":
    args = parse_args()
    args.cuda = not args.no_cuda
    args.d_word_vec = args.d_model

    # for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    if not args.output_dir:
        raise ValueError("No experiment result will be saved.\nPlease specify output directory")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.batch_size < 2048 and args.n_warmup_steps <= 4000:
        print("[Warning] The warmup steps may be not enough.\n"\
              "(sz_b, warmup) = (2048, 4000) is the official setting.\n"\
              "Using smaller batch w/o longer warmup may cause "\
              "the warmup stage ends with only little data trained.")

    device = torch.device("cuda" if args.cuda else "cpu")

    # preprocess step
    preprocessor_module = importlib.import_module(f"attention.preprocess.{args.language_model}").Preprocess
    preprocessor = preprocessor_module(**vars(args))
    preprocessor.preprocess_raw_data()

    # prepare data_loader
    print(args)
    training_data, validation_data, args = preprocessor.prepare_dataloader(args, device)

    # prepare trainer
    trainer_module = importlib.import_module(f"attention.train.{args.language_model}").Train
    trainer = trainer_module(
        model_name=args.language_model,
        epoch=args.epoch,
        arg=args,
        device=device
    )

    # train model
    trainer.train(training_data, validation_data)