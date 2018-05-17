import sys

def main():
    sys.stdin.readline()
    for line in sys.stdin:
        x = line.strip().split(',')
        word1, word2, ave = x[0], x[1], x[-1]
        print("{}\t{}\t{}".format(word1, word2, ave))

if __name__ == '__main__':
    main()
