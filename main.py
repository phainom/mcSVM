from getopt import getopt, GetoptError
from sys import argv, exit
from q1a import run as _q1a
from q1b import run as _q1b
from q1c import run as _q1c


def q1a():
    _q1a()


def q1b():
    _q1b()


def q1c():
    _q1c()


def main(argv):
    """
    To be called by correctors. Executes code demonstrating answers to
    specific problem sheet questions which can be specified through command line argument.
    """
    def usage():
        print """The following command line arguments are available:"
                 --help, -h: print usage info
                 -q 1a: run code for question 1a
                 -q 1b: run code for question 1b
                 -q 1c: run code for question 1c
              """
    try:
        opts, args = getopt(argv[1:], "hq:", ["help"])
    except GetoptError:
        usage()
        exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            exit()
        elif opt in ("-q",):
            if arg == "1a":
                print "Executing code for question 1a..."
                q1a()
            elif arg == "1b":
                print "Executing code for question 1b..."
                q1b()
            elif arg == "1c":
                print "Executing code for question 1c..."
                q1c()
            else:
                print "Question ", arg, " not available!"
                exit(2)

if __name__ == "__main__":
    main(argv)
