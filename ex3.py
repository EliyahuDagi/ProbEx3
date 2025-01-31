import sys
from utils import process_file, split_to_n, filter_rare_words
from em import initialize_EM


def main(dev_path):
    articles = process_file(dev_path)
    articles = filter_rare_words(articles, 3)

    alpha_i, p_word_given_x = initialize_EM(articles)
    #p_x, p_word_given_x, p_x_given_y, p_y = run_EM(articles)


if __name__ == '__main__':
    main(*sys.argv[1:])
