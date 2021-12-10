
def add_common_options(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )

def add_context_options(parser, with_home=True):
    common = parser.add_argument_group("context options")
    if with_home:
        common.add_argument('-H', '--home', type=str, default=".", help="The path to the mlonmcu environment (overwriting $MLONMCU_HOME environment variable)")

