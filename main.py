#Main

import yaml

def main():
    #Config
    with open("Configs/config.yaml") as f:
        config = yaml.safe_load(f)


if __name__ == "__main__":
    main()