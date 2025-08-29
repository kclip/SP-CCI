import argparse
from sp_cci.plotting import make_coverage_and_width_plots

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="JSON results files")
    ap.add_argument("--outdir", type=str, default="Figures")
    args = ap.parse_args()
    make_coverage_and_width_plots(args.inputs, args.outdir)

if __name__ == "__main__":
    main()
