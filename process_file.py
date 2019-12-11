#!/usr/bin/env python3

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-l","--log_file", metavar="N", type=str, help="logging file")
    parser.add_argument("-f","--file", metavar="N", type=str, help="recording file")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        verbose = True

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, filename="groundstation.log"
    )

    sys.exit(main())
