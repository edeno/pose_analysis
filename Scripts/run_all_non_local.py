import os
import subprocess
import sys
from argparse import ArgumentParser

from loren_frank_data_processing import make_epochs_dataframe
from src.parameters import ANIMALS


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_type', type=str, default='clusterless')
    parser.add_argument('--training_type', type=str,
                        default='no_ripple_and_no_ascending_theta')
    parser.add_argument('--overwrite', action='store_true')

    return parser.parse_args()


def run_bash(epoch_key, log_directory, args):
    animal, day, epoch = epoch_key
    print(f'Animal: {animal}, Day: {day}, Epoch: {epoch}')
    bash_cmd = (f'python decode_non_local_by_epoch.py {animal} {day} {epoch}'
                f' --data_type {args.data_type}'
                f' --training_type {args.training_type}'
                )
    if args.overwrite:
        bash_cmd += ' --overwrite'

    log_file = os.path.join(
        log_directory,
        f'{animal}_{day:02d}_{epoch:02d}_{args.data_type}_'
        f'{args.training_type}.log')

    with open(log_file, 'w') as file:
        try:
            subprocess.run(bash_cmd, shell=True, check=True,
                           stderr=subprocess.STDOUT, stdout=file)
        except subprocess.CalledProcessError:
            print(f'Error in {epoch_key}')


def main():
    log_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_directory,  exist_ok=True)

    args = get_command_line_arguments()

    epoch_info = make_epochs_dataframe(ANIMALS)
    epoch_info = epoch_info.loc[(epoch_info.type == 'run')]

    for epoch_key in epoch_info.index:
        run_bash(epoch_key, log_directory, args)


if __name__ == '__main__':
    sys.exit(main())
