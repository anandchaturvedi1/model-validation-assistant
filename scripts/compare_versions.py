"""Compare two simple text model docs and highlight differences (line-based)."""
import sys, difflib
def main(a,b):
    with open(a) as f: A=f.readlines()
    with open(b) as f: B=f.readlines()
    for line in difflib.unified_diff(A,B,fromfile=a,tofile=b,n=3):
        sys.stdout.write(line)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    args=p.parse_args()
    main(args.a,args.b)
