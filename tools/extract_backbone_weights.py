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
    # print(ck.keys())
    for key, value in ck['state_dict'].items():
    # for key, value in ck.items():
        print(key,key[7:])
        # output_dict['state_dict'][key] = value
        # has_backbone = True
        if key.startswith('backbone'):
            output_dict['state_dict'][key[9:]] = value
            has_backbone = True
        # if key.startswith('module.encoder_q') and not key.startswith('module.encoder_q.fc'):
        #     output_dict['state_dict'][key[17:]] = value
        # if key.startswith('module') and not key.startswith('module.fc'):
            # if  key.startswith('module.fc'):
            #     output_dict['state_dict']['fc_cls'+key[9:]] = value
        #     # else:
            # output_dict['state_dict'][key[7:]] = value
        # if not key.startswith('fc'):
        #     output_dict['state_dict'][key] = value
            # has_backbone = True

    # for key, value in ck['state_dict'].items():
    # for key, value in ck.items():
    #     if not key.startswith('fc'):
    #         output_dict['state_dict'][key] = value
    #         has_backbone = True
    if not has_backbone:
        raise Exception("Cannot find a backbone module in the checkpoint.")
    torch.save(output_dict, args.output)


if __name__ == '__main__':
    main()
