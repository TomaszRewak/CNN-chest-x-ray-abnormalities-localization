import sys
import urllib


def main(path, url):
    urllib.urlretrieve (url, path)


if __name__ == '__main__':
    main(sys.argv[1], 'https://s3.amazonaws.com/cadl/models/vgg16.tfmodel')


# python scraper/download_model.py data/vgg16.tfmodel
