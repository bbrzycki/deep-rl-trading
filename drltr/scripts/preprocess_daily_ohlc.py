import pandas as pd
import os


def split_df(input_fn, output_dir, split=0.9):
    df = pd.read_csv(input_fn).sort_values('Date')
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


def split_df2(input_fn, input_fn2, output_dir, split=0.9):
    df = pd.read_csv(input_fn).sort_values('Date')
    df2 = pd.read_csv(input_fn2).sort_values('Date')
    shared_length = min(len(df), len(df2))
    df = df.iloc[-shared_length:]
    df2 = df2.iloc[-shared_length:]

    train_df = df[:int(shared_length*split)]
    test_df = df[int(shared_length*split):]
    train_df2 = df2[:int(shared_length*split)]
    test_df2 = df2[int(shared_length*split):]

    base = os.path.basename(input_fn)
    head = os.path.splitext(base)[0]
    base2 = os.path.basename(input_fn2)
    head2 = os.path.splitext(base2)[0]

    if output_dir != '' and output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    train_fn = '{}/{}_pair_train.csv'.format(output_dir, head)
    train_df.to_csv(train_fn, index=False)
    print('Train data saved to {} ({} lines)'.format(train_fn, len(train_df)))

    test_fn = '{}/{}_pair_test.csv'.format(output_dir, head)
    test_df.to_csv(test_fn, index=False)
    print('Test data saved to {} ({} lines)'.format(test_fn, len(test_df)))

    train_fn2 = '{}/{}_pair_train.csv'.format(output_dir, head2)
    train_df2.to_csv(train_fn2, index=False)
    print('Train data saved to {} ({} lines)'.format(train_fn2, len(train_df2)))

    test_fn2 = '{}/{}_pair_test.csv'.format(output_dir, head2)
    test_df2.to_csv(test_fn2, index=False)
    print('Test data saved to {} ({} lines)'.format(test_fn2, len(test_df2)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--filename2', '-f2', type=str, default='')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('-s', '--split', type=float, default=0.9)

    args = parser.parse_args()

    params = vars(args)

    if params['filename2'] == '':
        split_df(params['filename'], params['output_dir'], params['split'])
    else:
        split_df2(params['filename'], params['filename2'], params['output_dir'], params['split'])


if __name__ == "__main__":
    main()
