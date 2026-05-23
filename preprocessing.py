import json
from collections import Counter
import random
import torch
import torchvision.transforms as transforms
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TRAIN_PART = 0.70
VALIDATION_PART = 0.15
TEST_PART = 0.15
CANAL_NORM_AVG = [0.485, 0.456, 0.406]
CANAL_NORM_STDS = [0.229, 0.224, 0.225]

classes_dict = {
    '["mug"]': 0,
    '["flat_plate"]': 1,
    '["soup_plate"]': 2,
    '["bowl"]': 3,
    '["pot"]': 4,
    '["wine_glass"]': 5,
    '["saucepan"]': 6,
}

def plot_class_distribution(train_set, validation_set, test_set):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(classes_dict.keys(), train_set.count([img for img in train_set if img[1] in classes_dict.keys()]))
    plt.title("Class Distribution in Train Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.bar(classes_dict.keys(), validation_set.count([img for img in validation_set if img[1] in classes_dict.keys()]))
    plt.title("Class Distribution in Validation Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.bar(classes_dict.keys(), test_set.count([img for img in test_set if img[1] in classes_dict.keys()]))
    plt.title("Class Distribution in Test Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    plt.savefig("class_distribution.png")

def plot_annotator_distribution(train_set_count, validation_set_count, test_set_count):
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(train_set_count.keys(), train_set_count.values())
    plt.title("Annotator Distribution in Train Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.bar(validation_set_count.keys(), validation_set_count.values())
    plt.title("Annotator Distribution in Validation Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.bar(test_set_count.keys(), test_set_count.values())
    plt.title("Annotator Distribution in Test Set")
    plt.xlabel("Annotator")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
    plt.savefig("annotator_distribution.png")

def extract_annotation(result_list):
    extracted = {}
    for r in result_list:
        from_name = r.get("from_name")
        val = r.get("value", {})
        
        if "choices" in val:
            extracted[from_name] = val["choices"]
        elif "text" in val:
            extracted[from_name] = val["text"]
    return extracted

def is_unclassifiable(annotation_dict):
    status = annotation_dict.get("image_status", [])
    return "multiple_different_classes" in status or "unreadable" in status

def process_annotations(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    accepted_images = []
    rejected_images = []

    for task in data:
        image_url = task.get("data", {}).get("image")
        annotations_data = task.get("annotations", [])
            
        parsed_annotations = [extract_annotation(ann.get("result", [])) for ann in annotations_data]
        num_people = len(parsed_annotations)
        
        if num_people == 1:
            ann = parsed_annotations[0]
            if is_unclassifiable(ann):
                rejected_images.append(image_url)
            else:
                id = ''
                if 'object_id' in ann:
                    id = ann['object_id']
                accepted_images.append({"image": image_url, "final_annotation": ann, "num_annotators": num_people, "id": id})
                
        else:
            statuses = [a['image_status'] for a in parsed_annotations]
            if statuses.count(['unreadable']) + statuses.count(['multiple_different_classes']) > 1:
                rejected_images.append(image_url)
            else:
                counter_anns = [json.dumps(a['dish_class']) if 'dish_class' in a else '' for a in parsed_annotations]
                counter = Counter(counter_anns)
                if len(counter) == num_people:
                    rejected_images.append(image_url)
                else:
                    id = ''
                    for parsed_ann in parsed_annotations:
                        if 'object_id' in parsed_ann:
                            id = parsed_ann['object_id']
                            break
                    accepted_images.append({"image": image_url, "final_annotation": counter.most_common(1)[0][0], "num_annotators": num_people, "id":id})

    return accepted_images, rejected_images
    

if __name__ == "__main__":
    file_name = "project-3-at-2026-04-26-12-35-909a11c3.json"
    
    # Przetwarzanie zdjęć ze względu na przydzielone im adnotacje; ze zbioru usuwane są te oznaczone jako nieklasyfikowalne i takie, w których nie ma większości głosów na daną klasę
    accepted, rejected = process_annotations(file_name)
    
    print(f"Zaakceptowane zdjęcia: {len(accepted)}")
    print(f"Odrzucone zdjęcia: {len(rejected)}")

    #Wykres liczby zaakceptowanych i odrzuconych zdjęć
    plt.bar(['Accepted', 'Rejected'], [len(accepted), len(rejected)], color=['green', 'red'])
    plt.title("Count of Accepted and Rejected Images")
    plt.savefig("accepted_rejected_count.png")

    image_set = {
        "m": [],
        "a": [],
        "i": [],
        "w": [],
        "no_id": []
    }
    annotation_set = {
        "m": [],
        "a": [],
        "i": [],
        "w": [],
        "no_id": []
    }
    
    for image in accepted:
        #Zdjęcia są przekształcane do rozmiaru 224x224 (standardyzacja)
        target_size = (224, 224)
        image_cv2 = cv2.imread(image["image"])
        resized_image = cv2.resize(image_cv2, target_size)
        
        transform = transforms.Compose([transforms.ToTensor()])
        input_tensor = transform(resized_image)

        #Podział zdjęć ze względu na adnotatorów, którzy je oznaczyli, dla dalszego zachowania proporcji adnotatorów w zbiorach treningowym, walidacyjnym i testowym; osobne zbiory na dane oznaczane wspólnie i bez id
        if image["num_annotators"] != 1:
            image_set["w"].append([input_tensor, image["final_annotation"], image["id"]])
        else:
            who_annotated = image["num_annotators"][0] if image["num_annotators"] != '' else "no_id"
            image_set[who_annotated].append([input_tensor, image["final_annotation"], image["id"]])

    #Augmentacja danych - tworzenie nowych, zaszumionych obrazów
    for key in image_set.keys(): 
        if key == "no_id" or key == "w":
            continue
        for i in range(50):
            random_index = torch.randint(0, len(image_set[key]), (1,)).item()
            original_image, original_annotation, original_id = image_set[key][random_index]
            noise = torch.randn_like(original_image)*0.01
            augmented_image = original_image + noise
            image_set[key].append([augmented_image, original_annotation, original_id])

    #Normalizacja danych do przedziału [-1, 1] - średnie i odchylenia standardowe dla poszczególnych kanałów są widoczne poniżej
    for key in image_set.keys():
        for image in image_set[key]:
            normalize = transforms.Normalize(mean=CANAL_NORM_AVG, std=CANAL_NORM_STDS)
            transform = transforms.Compose([normalize])
            image[0] = transform(image[0])
    
    #printing zdjęcie, do wyrzucenia potem
    first_image_tensor = image_set['m'][0][0]
    first_image_tensor = first_image_tensor.permute(1, 2, 0).numpy()
    first_image_tensor = (first_image_tensor * 255).astype('uint8')
    cv2.imwrite(f"first_image.jpg", first_image_tensor)

    train_set, validation_set, test_set = [], [], []
    train_set_annotations, validation_set_annotations, test_set_annotations = [], [], []

    #Dicts only for plotting annotator distribution later
    train_set_count = {
        "m": 0,
        "a": 0,
        "i": 0,
    }

    validation_set_count = {
        "m": 0,
        "a": 0,
        "i": 0,
    }

    test_set_count = {
        "m": 0,
        "a": 0,
        "i": 0,
    }

    #Podział zbiorów z zachowaniem:
    # - proporcji klas
    # - proporcji adnotatorów
    # i zapobieganiem wyciekom danych (wszystkie zdjęcia z danym id trafiają do tego samego zbioru)
    for key in image_set.keys():
        if key == "no_id":
            continue
        for classname in classes_dict.keys():
            class_count = image_set[key].count([img for img in image_set[key] if img[1] == classname])
            id_dict = {}
            for img in image_set[key]:
                if img[1] == classname:
                    id = img[2]
                    if id not in id_dict:
                        id_dict[id] = 0
                    id_dict[id] += 1
            
            chosen_count = 0
            while chosen_count < class_count*TRAIN_PART:
                id = random.choice(list(id_dict.keys()))
                if id_dict[id] > 0:
                    for i in range(len(image_set[key])):
                        if image_set[key][i][2] == id:
                            train_set.append(image_set[key][i])
                    chosen_count += id_dict[id]
                    train_set_count[key] += id_dict[id]
                    del image_set[key][i]
                    del id_dict[id]

            chosen_count = 0
            while chosen_count < class_count*VALIDATION_PART:
                id = random.choice(list(id_dict.keys()))
                if id_dict[id] > 0:
                    for i in range(len(image_set[key])):
                        if image_set[key][i][2] == id:
                            validation_set.append(image_set[key][i])
                    validation_set_count[key] += id_dict[id]
                    chosen_count += id_dict[id]
                    del image_set[key][i]
                    del id_dict[id]

            test_set.extend(image_set[key])
            test_set_count[key] += len(image_set[key])

    train_set_no_id_with_val, test_set_no_id = train_test_split(image_set["no_id"], test_size=TEST_PART, random_state=42)
    train_set_no_id, validation_set_no_id = train_test_split(train_set_no_id_with_val, test_size=VALIDATION_PART, random_state=42)

    train_set.extend(train_set_no_id)
    validation_set.extend(validation_set_no_id)
    test_set.extend(test_set_no_id)

    #Przemieszanie zbiorów
    random.shuffle(train_set)
    random.shuffle(validation_set)
    random.shuffle(test_set)

    #Przygotowanie wektorów adnotacji dla zdjęć
    for image in train_set:
        annotation_vector = torch.zeros(len(classes_dict))
        annotation_vector[classes_dict[image["final_annotation"]]] = 1
        train_set_annotations.append(annotation_vector)

    for image in validation_set:
        annotation_vector = torch.zeros(len(classes_dict))
        annotation_vector[classes_dict[image["final_annotation"]]] = 1
        validation_set_annotations.append(annotation_vector)

    for image in test_set:
        annotation_vector = torch.zeros(len(classes_dict))
        annotation_vector[classes_dict[image["final_annotation"]]] = 1
        test_set_annotations.append(annotation_vector)
    
    #Wykreśl statystyki
    plot_class_distribution(train_set, validation_set, test_set)
    plot_annotator_distribution(train_set_count, validation_set_count, test_set_count)
    
