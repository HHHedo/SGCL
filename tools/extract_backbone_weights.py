import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output', type=str, help='destination file name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output.endswith(".pth")
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    output_dict = dict(state_dict=dict(), author="DenseCL")
    has_backbone = False
    for key, value in ck['state_dict'].items():
        # print(key,key[7:])
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
        # if key.startswith('module.encoder') and not key.startswith('module.encoder.fc'):
            # output_dict['state_dict'][key[15:]] = value
        # if key.startswith('module'):
        #     if  key.startswith('module.fc'):
        #         output_dict['state_dict']['fc_cls'+key[9:]] = value
        #     else:
        #         output_dict['state_dict'][key[7:]] = value
            has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
