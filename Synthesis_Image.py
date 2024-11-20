import os
import random
import json
import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from fontTools.ttLib import TTFont
from agument_data import CustomTransform
from multiprocessing import Pool, cpu_count

class Agument_method:
    def __init__(self):
        self.aug_method = CustomTransform()

    def augment_fuct(self, img):
        return self.aug_method(img)

def find_general_vocab(vocab):
    with open('/home/tuandao/download-hanom/data/meta.json', 'r') as f:
        data_json = json.load(f)
        vocab_general = data_json.values()
    return list(set(vocab) & set(vocab_general))

def resize_to_min_size(img, min_size):
    original_width, original_height = img.size
    if original_width <= original_height:
        scale_factor = min_size / original_width
    else:
        scale_factor = min_size / original_height
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    img = img.resize((new_width, new_height))
    return img

def extract_font_vocab(font_path):
    font = TTFont(font_path)
    cmap = font['cmap']
    char_set = set()
    other_vocab = ['\x00', '\x08', '\t', '\r', '\x1d']
    for table in cmap.tables:
        char_set.update(chr(c) for c in table.cmap.keys())
    return char_set

def generate_image(args):
    image_num, font_path, vocabularies, list_bg_image, image_dir, text_dir = args
    augmentor = Agument_method()

    # Load background image
    background_img = Image.open(random.choice(list_bg_image))
    background_img = resize_to_min_size(background_img, random.randint(480, 640))

    draw = ImageDraw.Draw(background_img)
    image_width, image_height = background_img.size
    font_size = random.randint(int(min(image_width, image_height) / 20), int(min(image_width, image_height) / 10))
    start_offset = random.randint(font_size + int(min(image_width, image_height) / 100), font_size + int(min(image_width, image_height) / 20))

    # Define font and size
    font = ImageFont.truetype(font_path, font_size)
    line_num = int(image_width / font_size) - 1

    chinese_lines = []
    for line in range(line_num):
        if random.random() < 0.6:
            char_num = random.randint(1, int(image_height / font_size) - 1)
        else:
            char_num = random.randint(int(image_height / font_size) * 3 // 4 - 1, int(image_height / font_size) - 1)
        list_char = [random.choice(vocabularies) for _ in range(char_num)]
        line_text = "".join(list_char)
        chinese_lines.append(line_text)

    # Set initial starting position
    x_position = image_width - start_offset
    line_spacing = (font_size * 1.1)

    # Iterate through each line of characters
    for line in chinese_lines:
        y_position = start_offset - font_size + random.randint(1, font_size // 2)
        for char in line:
            draw.text((x_position, y_position), char, font=font, fill="black")
            y_position += font_size

        x_position -= line_spacing
        if x_position < font_size * 0.3:
            break

    # Augment the image with a probability of 0.5
    if random.random() < 0.5:
        background_img = augmentor.augment_fuct(background_img.convert('RGB'))

    # Save the image
    image_filename = os.path.join(image_dir, f"image_{image_num:04d}.jpg")
    background_img = background_img.convert("RGB")
    background_img.save(image_filename)

    # Save the text lines to a file
    text_filename = os.path.join(text_dir, f"image_{image_num:04d}.txt")
    with open(text_filename, 'w', encoding='utf-8') as text_file:
        text_file.write("\n".join(chinese_lines))

def main():
    image_dir = "generated_images"
    text_dir = image_dir
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    font_paths = [
        '/home1/vudinh/NomNaOCR/VLM/Synthesis_Code/fonts/pysvg2font_sample.ttf',
    ]
    font_path = font_paths[0]
    list_bg_image = glob.glob("/home1/vudinh/NomNaOCR/VLM/Synthesis_Code/backgrounds/*")
    vocabularies = find_general_vocab(list(extract_font_vocab(font_path)))

    num_images = 100  # Adjust the number of images to generate
    num_processes = min(cpu_count(), 4)  # Use available cores, limit to 8
    args = [
        (image_num, font_path, vocabularies, list_bg_image, image_dir, text_dir)
        for image_num in range(num_images)
    ]

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(generate_image, args), total=num_images))

if __name__ == "__main__":
    main()
