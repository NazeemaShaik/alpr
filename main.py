import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", help='path to image')
parser.add_argument("-t", help="tesseract only", action="store_true")
parser.add_argument("-v", help="keras OCR + easy OCR", action="store_true")
parser.add_argument("-a", help="run all", action="store_true")
parser.add_argument("--verbose", help='verbose', action="store_true")
args = parser.parse_args()
IMAGE_PATH = args.image


def execute_process_ocr(image_path, verbose):
    command = f"python3 process_ocr.py --image {image_path}"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, err = process.communicate()
    if verbose:
        print("Output:", "-"*10)
        print(output)
        print("Error:", "-"*10)
        print(err)


def execute_vanilla_ocr(image_path, verbose):
    command = f"python3 vanilla_ocr.py --image {image_path}"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, err = process.communicate()
    if verbose:
        print("Output:", "-"*10)
        print(output)
        print("Error:", "-"*10)
        print(err)
    


def main():
    if (args.t == False and args.v == False and args.a == False) or args.a == True:
        execute_process_ocr(IMAGE_PATH, args.verbose)
        execute_vanilla_ocr(IMAGE_PATH, args.verbose)

    elif args.t == True:
        execute_process_ocr(IMAGE_PATH, args.verbose)

    elif args.v == True:
        execute_vanilla_ocr(IMAGE_PATH, args.verbose)

    else:
        raise NotImplementedError("Wrong Flag")


if __name__ == "__main__":
    main()
