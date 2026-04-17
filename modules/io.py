import yaml
import csv


def prompt_continue(message="Wish to continue? [y/n]: "):
    """Prompt until a valid y/n answer is provided."""
    while True:
        answer = input(message).strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please answer with 'y' or 'n'.")


def read_lines(fn):
    f = open(fn,'r').readlines()
    f = [s.replace('\n','') for s in f]
    return f


def load_yaml(fn):
    """loads a yaml file into a dict"""
    with open(fn,'r') as file_:
        try:
            return yaml.load(file_, Loader=yaml.loader.SafeLoader)
        except RuntimeError as e:
            print("failed to load yaml fille {}, {}\n".format(fn,e))

def save_yaml(fn, data):
    with open(fn,'w') as file_:
        yaml.dump(data,file_, default_flow_style=False)

def write_csv(filename,dict):
    with open(filename,'w') as f:
        w = csv.DictWriter(f,dict.keys())
        w.writeheader()
        w.writerow(dict)

def read_csv(filename):
    with open(filename,'r') as f:
        w = csv.DictReader(f)
        d = [row for row in w]
        if len(d)==1:
            return d[0]
        else:
            return d
