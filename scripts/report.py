"""Generate a model report on multiple sessions from the command line.

> python /path/to/report.py
> --save_dir=/path/to/save_dir
> --model_dir=/path/to/model_dir
> --features_dir=/path/to/features_dir
> --videos_dir=/path/to/videos_dir
> --format=pdf

"""

import argparse
from daart_utils.reports import ReportGenerator


def run():

    pass

    # args = parser.parse_args()
    #
    # reporter = ReportGenerator(
    #
    # )
    # output_dir = reporter.generate_report(
    #     save_dir=save_dir, format=format, video_kwargs={"max_frames": 100},
    # )
    # print('Diagnostics saved to %s' % output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--features_dir')
    parser.add_argument('--videos_dir', default=None)
    parser.add_argument('--format', default='pdf')
    parser.add_argument('--framerate', default=None)

    run()
