import sys

def main(dataset_path):
    with open(dataset_path, "r") as f:
        lines = f.readlines()
    lines = [l.split() for l in lines]
    filenames = [l[0] for l in lines]
    tokens = [l[1:66] for l in lines]

    for ti in range(len(tokens)):
        for tj in range(ti + 1, len(tokens)):
            corr = sum(a == b for a, b in zip(tokens[ti], tokens[tj]))
            if corr > 30 and not (filenames[ti].startswith(filenames[tj]) or filenames[tj].startswith(filenames[ti])) and len(set(tokens[ti])) >= 10:
                print(filenames[ti], filenames[tj], corr)


if __name__ == "__main__":
    main(sys.argv[1])
