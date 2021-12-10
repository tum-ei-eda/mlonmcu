from ..tvm_flow import get_parser

from .backend import TVMBackend

class TVMAOTBackend(TVMBackend):
    
    def generate_code(self):
        pass

def main():
    parser = get_parser()
    print("PARSER", parser, parser._actions)
    parser.parse_args()
    pass

if __name__ == "__main__":
    main()
