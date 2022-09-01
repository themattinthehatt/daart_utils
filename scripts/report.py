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

    args = parser.parse_args()

    reporter = ReportGenerator(model_dir=args.model_dir)
    reporter.generate_report(
        save_dir=args.save_dir,
        features_dir=args.features_dir,
        format=args.format,
        bout_example_kwargs={
            # 'features_to_plot': ['orientation', 'operculum', 'oper_angle_avg', 'oper_dist_avg'],
            'frame_win': 200,
            'max_n_ex': 10,
            'min_bout_len': 5,
        },
        video_kwargs={
            'max_frames': 500,
            'framerate': 20,
        },
        video_framerate=float(args.framerate),
        videos_dir=args.videos_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--features_dir')
    parser.add_argument('--videos_dir', default=None)
    parser.add_argument('--format', default='pdf')
    parser.add_argument('--framerate', default=None)

    run()
