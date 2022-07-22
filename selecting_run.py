import os
import argparse
from glob import glob
from os.path import join


def main(args):
    style1_path_list, style2_path_list, content_path_list = [], [], []
    for d in args.input_dir_num:
        img_path = glob(join('input', d, '*.png'))
        for p in img_path:
            if 'style1' in p:
                style1_path_list.append(p)
            elif 'style2' in p:
                style2_path_list.append(p)
            elif 'content' in p:
                content_path_list.append(p)

    for style1_path, style2_path, content_path in zip(style1_path_list, style2_path_list, content_path_list):
        command = f'pipenv run python st.py \
                    -serif_style_path {style1_path} \
                    -nonserif_style_path {style2_path} \
                    -content_path {content_path} \
                    -output_path {args.output_dir} \
                    -sw1 {args.style_weight[0]} \
                    -sw2 {args.style_weight[1]} \
                    -sw3 {args.style_weight[2]} \
                    -sw4 {args.style_weight[3]} \
                    -sw5 {args.style_weight[4]} \
                    -cw1 {args.content_weight} \
                    -cew1 {args.cross_entropy_weight}'

        print(command)
        os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', help='Output dir path', type=str, default='output/')
    parser.add_argument('-i', '--input_dir_num', nargs='*', help='Input directory e.x. (-i 0, 1, 2 ...)', type=str, default=None)
    parser.add_argument('-sw', '--style_weight', nargs='*', help='Style weights e.x. (-sw 1 1 1 1 1)', type=float, default=[1,1,1,1,1])
    parser.add_argument('-cw', '--content_weight', help='Content weights e.x. (-cw 1)', type=float, default=1)
    parser.add_argument('-cew', '--cross_entropy_weight', help='Cross Entropy weights e.x. (-cew 1)', type=float, default=1e5)
    args = parser.parse_args()

    if args.input_dir_num is None:
        input_dir_num = len(glob(join('input', '*')))
        args.input_dir_num = [str(i) for i in range(1, input_dir_num+1)]

    main(args)
