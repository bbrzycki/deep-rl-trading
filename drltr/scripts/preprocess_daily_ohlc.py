import pandas as pd
import os


def split_df(input_fn, output_dir, split=0.9):
    df = pd.read_csv(input_fn)
    df = df.sort_values('Date')
    train_df = df[:int(len(df)*split)]
    test_df = df[int(len(df)*split):]

    base = os.path.basename(input_fn)
    head = os.path.splitext(base)[0]
    if output_dir != '' and output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    train_fn = '{}/{}_train.csv'.format(output_dir, head)
    train_df.to_csv(train_fn, index=False)
    print('Train data saved to {} ({} lines)'.format(train_fn, len(train_df)))

    test_fn = '{}/{}_test.csv'.format(output_dir, head)
    test_df.to_csv(test_fn, index=False)
    print('Test data saved to {} ({} lines)'.format(test_fn, len(test_df)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('-s', '--split', type=float, default=0.9)

    args = parser.parse_args()

    params = vars(args)

    split_df(params['filename'], params['output'], params['split'])


if __name__ == "__main__":
    main()
