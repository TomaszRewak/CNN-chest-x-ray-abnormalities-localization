from lxml import html
import requests
import re
import json
import urllib
import sys
import os


def get_table_url(domain, page):
    return '{0}gridquery.php?q=&it=x,xg&sub=x&m={1}&n={2}'.format(
        domain,
        1 + 100 * page,
        100 + 100 * page
    )


def get_page_urls(domain, table_url):
    request = requests.get(table_url)
    tree = html.fromstring(request.text)

    script = tree.xpath('//script[@language="javascript"]/text()')[0]

    json_string = re.compile(r"var oi = (.*);").findall(script)[0]
    json_data = json.loads(json_string)

    return [
        '{0}{1}'.format(domain, x['nodeRef'])
        for x in json_data
    ]


def extract_info(domain, url):
    request = requests.get(url)

    dom = html.fromstring(request.text)
    dom_labels = dom.xpath(
        '//table[@class="masterresultstable"]//div[@class="meshtext-wrapper-left"]'
    )

    if dom_labels != []:
        dom_labels = dom_labels[0]
    else:
        return None

    image_data = {}

    image_data['type'] = dom_labels.xpath('.//strong/text()')[0]
    image_data['items'] = dom_labels.xpath('.//li/text()')
    image_data['url'] = domain + dom.xpath('//img[@id="theImage"]/@src')[0]
    image_data['name'] = os.path.basename(image_data['url'])

    print(image_data)

    return image_data


def download_image(basePath, info):
    urllib.urlretrieve(
        info['url'],
        os.path.join(basePath, info['name'])
    )


def main(path, domain):
    info = [
        extract_info(domain, page_url)
        for page in range(0, 75)
        for page_url in get_page_urls(domain, get_table_url(domain, page))
    ]

    info = [
        extracted_info
        for extracted_info in info
        if extracted_info is not None
    ]

    with open(os.path.join(path, 'images-description.json'), 'w') as f:
        json.dump(info, f, indent=4, separators=(',', ': '))

    images_path = os.path.join(path, 'images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    print("Downloading images")

    for value in info:
        download_image(images_path, value)


if __name__ == '__main__':
    main(sys.argv[1], 'https://openi.nlm.nih.gov/')


# python scraper/scraper.py data
