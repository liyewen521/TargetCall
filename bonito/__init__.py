from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__version__ = '0.5.0'


def main():
    from bonito.cli import basecaller

    parser = ArgumentParser(
        'bonito',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    basecaller_parser = subparsers.add_parser(
        'basecaller',
        parents=[basecaller.argparser()],
    )
    basecaller_parser.set_defaults(func=basecaller.main)

    args = parser.parse_args()
    args.func(args)
