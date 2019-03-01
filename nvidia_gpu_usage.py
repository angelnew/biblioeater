import os


# insert in intermediate steps of the process if you use GPU(s) and it gets out of memory
def get_gpu_usage():

    os.system("nvidia-smi >smi.txt")

    with open("smi.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    items = lines[8].split("|")[2].split("/")

    return (items[0][:-4], items[1][:-4])


if __name__ == "__main__":
    usage = get_gpu_usage()
    print(usage)