import argparse
import luigi
import sys
from predictor import Predictor


parser = argparse.ArgumentParser()
parser.add_argument('--input-path', help='Input path containing files',
                    default='.')
parser.add_argument('--output-path', help='Directory where output file will be generated',
                    default='.')


def main():
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    task = Predictor(input_path=args.input_path,
                      output_path=args.output_path)

    if luigi.build([task], local_scheduler=True, workers=1):
        print(task.output())

if __name__ == '__main__':
    main()
