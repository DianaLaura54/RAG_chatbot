import fitz
import os
from PIL import Image
from io import BytesIO


def convert_image_format(image_bytes, target_format='PNG'):
    ##image is a PIL Image object
    ##img_byte_array is a io.BytesIO() object
    ##image_bytes -  raw binary data of an image
    image = Image.open(BytesIO(image_bytes))
    img_byte_array = BytesIO()
    image.save(img_byte_array, format=target_format)
    return img_byte_array.getvalue()


def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    image_paths = []
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    # create a subfolder for this specific PDF
    pdf_subfolder = os.path.join(output_folder, pdf_filename)
    os.makedirs(pdf_subfolder, exist_ok=True)
    # pdf_path is a pdf file
    # iterate through the pages from the pdf
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        # iterate through the pdf file, take the index and the image from the image_list(the images from that page)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            if ext.lower() == 'jpx':
                print(f"Converting JPX image from {pdf_filename}_page_{page_num + 1}_img_{img_index + 1}.jpx")
                # convert the jpx image to png
                image_bytes = convert_image_format(image_bytes, 'PNG')
                ext = 'png'
            #use simple naming format that matches the display code expectations
            image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{ext}"
            # save in the PDF-specific subfolder
            image_path = os.path.join(pdf_subfolder, image_filename)
            # write it to the disk
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            # append it to the image_paths array, when you return all the images
            image_paths.append(image_path)
            print(f"Saved: {image_path}")
    doc.close()
    return image_paths


def process_pdf_folder(input_folder, output_base_folder="extracted_images", recursive=False):
    all_results = {}
    # create the directory,check if it exists
    os.makedirs(output_base_folder, exist_ok=True)
    # go through the input folder, where the pdfs are located, check the root and the files
    for root, _, files in os.walk(input_folder):
        for file in files:
            # check if the file is a pdf
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                try:
                    print(f"\nProcessing: {pdf_path}")
                    extracted_images = extract_images_from_pdf(pdf_path, output_base_folder)
                    # append the images in an array
                    all_results[pdf_path] = extracted_images
                    print(f"Extracted {len(extracted_images)} images from {file}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    all_results[pdf_path] = []
        if not recursive:
            break
    # return all the images from all the created folders
    return all_results

if __name__ == "__main__":
    input_folder = "Contents\\books"
    output_folder = "extracted_images"
    results = process_pdf_folder(input_folder, output_folder, recursive=True)
    total_images = sum(len(images) for images in results.values())
    print(f"\nTotal images extracted: {total_images}")