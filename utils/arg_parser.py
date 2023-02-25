import argparse


def get_argparser():

    parser = argparse.ArgumentParser("COTR parser", add_help=False)

    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument(
        '--data_path',
        default='/storage/datasets/fsc147',
        type=str
    )
    parser.add_argument(
        '--model_path',
        default='',
        type=str
    )
    parser.add_argument('--dataset', default='fsc147', type=str)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--swav_backbone', action='store_true')
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_dec_layers', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--normalized_l2', action='store_true')
    parser.add_argument('--count_loss_weight', default=0, type=float)
    parser.add_argument('--min_count_loss_weight', default=0, type=float)
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--use_query_pos_emb', action='store_true')
    parser.add_argument('--use_objectness', action='store_true')
    parser.add_argument('--use_appearance', action='store_true')
    parser.add_argument('--orig_dmaps', action='store_true')

    parser.add_argument('--skip_cars', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--fcos_pred_size', default=64, type=float)

    return parser
