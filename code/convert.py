from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("vec_name", nargs='?', default='embeddings_total.vec', help="Name of the embedding file to use")
    parser.add_argument("filtered_vec_name", nargs='?', default='embeddings.vec', help="Name of the filtered embedding file")

    return parser.parse_args()

def convert(resource_folder, vec_name, filtered_vec_name):

    # Read all the lines of the embedding file and save if containing babelnet
    lines = []
    with open(resource_folder + vec_name) as f:
        n_lines, embedding = f.readline().split()
        for line in f:
            if ':' in line:
                lines.append(line)

    # Record to new file
    with open(resource_folder + filtered_vec_name, "w+") as f:
        f.writelines(str(len(lines)) + ' ' + embedding + '\n')
        f.writelines(lines)

if __name__ == '__main__':
    args = parse_args()

    # Filter vec file
    convert(resource_folder=args.resource_folder, vec_name=args.vec_name, filtered_vec_name=args.filtered_vec_name)