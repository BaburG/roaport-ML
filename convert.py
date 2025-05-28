import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, output_folder, class_name="pothole"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(xml_folder):
        if file.endswith(".xml"):
            tree = ET.parse(os.path.join(xml_folder, file))
            root = tree.getroot()
            image_width = int(root.find("size/width").text)
            image_height = int(root.find("size/height").text)
            txt_filename = os.path.splitext(file)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)

            with open(txt_path, "w") as f:
                for obj in root.findall("object"):
                    cls = obj.find("name").text
                    if cls != class_name:
                        continue
                    bbox = obj.find("bndbox")
                    xmin = int(bbox.find("xmin").text)
                    ymin = int(bbox.find("ymin").text)
                    xmax = int(bbox.find("xmax").text)
                    ymax = int(bbox.find("ymax").text)

                    x_center = ((xmin + xmax) / 2) / image_width
                    y_center = ((ymin + ymax) / 2) / image_height
                    w = (xmax - xmin) / image_width
                    h = (ymax - ymin) / image_height

                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    convert_voc_to_yolo("annotations/", "labels/")