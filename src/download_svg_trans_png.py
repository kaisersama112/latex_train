import os
import re
import time
import requests
import cairosvg
from io import BytesIO
from PIL import Image
import threading


def add_default_size_to_svg(svg_content):
    width_pattern = re.compile(r'width\s*=\s*"([^"]+)"')
    height_pattern = re.compile(r'height\s*=\s*"([^"]+)"')
    has_width = width_pattern.search(svg_content)
    has_height = height_pattern.search(svg_content)
    if not has_width or not has_height:
        svg_content = re.sub(r'<svg\s', '<svg width="100" height="100" ', svg_content, count=1)

    return svg_content


def convert_svg_to_image(svg_path, output_path, format='PNG', background_color=(255, 255, 255)):
    if svg_path:
        with open(svg_path, 'r', encoding="utf-8") as file:
            svg_content = file.read()
        svg_content_with_size = add_default_size_to_svg(svg_content)
        png_data = cairosvg.svg2png(bytestring=svg_content_with_size.encode())
        image = Image.open(BytesIO(png_data))
        background = Image.new('RGBA', image.size, background_color + (255,))
        background.paste(image, (0, 0), image)
        background.save(output_path, format=format)


def download_and_convert_svg(url, svg_path, png_path):
    try:
        file_name = url.split('/')[-1]
        svg_file_path = os.path.join(svg_path, file_name)
        response = requests.get(url)
        response.raise_for_status()
        with open(svg_file_path, 'wb') as file:
            file.write(response.content)
        print(f"下载成功: {svg_file_path}")

        png_file_path = os.path.join(png_path, file_name.replace(".svg", ".png"))
        convert_svg_to_image(svg_file_path, png_file_path)

    except requests.RequestException as e:
        print(f"下载失败: {url}，错误: {e}")


def svg2png(data, svg_path, png_path):
    for url in data:
        download_and_convert_svg(url, svg_path, png_path)


def freeze_support_main(data_list, save_path="../svg_image", batch_size=512):
    svg_path = os.path.join(save_path, "svg")
    os.makedirs(svg_path, exist_ok=True)
    png_path = os.path.join(save_path, "png")
    os.makedirs(png_path, exist_ok=True)
    svg2png(data_list, svg_path, png_path)
    # threads = []
    # for i in range(0, len(data_list), batch_size):
    #     data_batch = data_list[i:i + batch_size]
    #     thread = threading.Thread(target=svg2png, args=(data_batch, svg_path, png_path))
    #     threads.append(thread)
    #     thread.start()
    #
    # for thread in threads:
    #     thread.join()


data = []
with open(r"D:\geren_test\haotiku\svgUrls.txt", "r") as file:
    for line in file.readlines():
        data.append(line.strip())

if __name__ == '__main__':
    """
    下载svgUrls.txt里面的svg图片并转换为png
    """
    freeze_support_main(data)
