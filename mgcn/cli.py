import argparse

def get_args():
    parser = argparse.ArgumentParser(description='MGCN for Offensive Language Detection')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layers', nargs='+', type=str, default=['en', 'de'],
                        help='Layers in MGCN.')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Sample size.')
    parser.add_argument('--method', type=str, default='bidict',
                        help='Method to compute edges between layers.')
    parser.add_argument('--model', type=str, default='bert',
                        help='Model to use as node features.')
    parser.add_argument('--finetune', type=bool, default=False,
                        help='Whether to finetune the model.')
    parser.add_argument('--batch_size', type=float, default=128,
                        help='Batch size.')

    return vars(parser.parse_args())
