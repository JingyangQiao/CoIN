import os
import glob
import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_vqa_texts_from_json_train(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_values = []

    for entry in data:
        if 'conversations' in entry:
            conversation = entry["conversations"][0]
            if 'value' in conversation:
                value = conversation['value']
                last_part = value.split('\n')[-1]
                extracted_values.append(last_part.strip())

    return extracted_values


def extract_ref_texts_from_json_train(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_values = []

    for entry in data:
        if 'conversations' in entry:
            conversation = entry["conversations"][0]
            if 'value' in conversation:
                value = conversation['value']
                last_part = value.split('\n')[-1].split(':')[0]
                extracted_values.append(last_part.strip())

    return extracted_values


def extract_imn_texts_from_json_train(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_values = []

    for entry in data:
        if 'conversations' in entry:
            conversation = entry["conversations"][0]
            if 'value' in conversation:
                value = conversation['value']
                last_part = value.split('\n')[1]
                extracted_values.append(last_part.strip())

    return extracted_values


def extract_vqa_texts_from_json_test(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_texts = []

    for entry in data:
        if 'text' in entry:
            text = entry['text']
            last_part = text.split('\n')[-1]
            extracted_texts.append(last_part.strip())

    return extracted_texts


def extract_ref_texts_from_json_test(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_texts = []

    for entry in data:
        if 'text' in entry:
            text = entry['text']
            last_part = text.split('\n')[-1].split(':')[0]
            extracted_texts.append(last_part.strip())

    return extracted_texts


def extract_imn_texts_from_json_test(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    extracted_texts = []

    for entry in data:
        if 'text' in entry:
            text = entry['text']
            last_part = text
            extracted_texts.append(last_part.strip())

    return extracted_texts


def get_instruction(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def select_lora(args):
    instructions = {}
    instruct_task = args.instruct.split("/")[-2]
    instruct_type = args.instruct.split("/")[-3]
    codebooks = glob.glob(args.codebook + "*.txt")
    if codebooks == []:
        if not os.path.exists(args.codebook): 
            print("This is the first task, we need to strat with a new set of LoRA.")
            os.makedirs(args.codebook)
        file_path = args.codebook + "instruction_1.txt"
        if instruct_task == "ImageNet":
            cur_instruct = list(set(extract_imn_texts_from_json_train(args.instruct)))
        elif instruct_task == "Grounding":
            cur_instruct = list(set(extract_ref_texts_from_json_train(args.instruct)))
        else:
            cur_instruct = list(set(extract_vqa_texts_from_json_train(args.instruct)))
        with open(file_path, 'a', encoding='utf-8') as file:
            for item in cur_instruct:
                file.write(f"{item}\n")
        return 
    else:
        for codebook in codebooks:
            order = codebook.split("/instruction_")[-1].split(".txt")[0]
            instruction = get_instruction(codebook)
            instructions[order] = instruction
    
    if "train" in args.instruct: 
        if instruct_task == "ImageNet":
            cur_instruct = list(set(extract_imn_texts_from_json_train(args.instruct)))
        elif instruct_task == "Grounding":
            cur_instruct = list(set(extract_ref_texts_from_json_train(args.instruct)))
        else:
            cur_instruct = list(set(extract_vqa_texts_from_json_train(args.instruct)))
    else:
        if instruct_task == "ImageNet":
            cur_instruct = list(set(extract_imn_texts_from_json_test(args.instruct)))
        elif instruct_task == "Grounding":
            cur_instruct = list(set(extract_ref_texts_from_json_test(args.instruct)))
        else:
            cur_instruct = list(set(extract_vqa_texts_from_json_test(args.instruct)))
    print(cur_instruct)
    orders = instructions.keys()
    sims = []
    for order in range(len(orders)):
        hist_instruct = instructions[str(order + 1)]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix_list = []
        for i in range(len(cur_instruct)):
            for j in range(len(hist_instruct)):
                tfidf_matrix = vectorizer.fit_transform([cur_instruct[i], hist_instruct[j]])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                tfidf_matrix_list.append(similarity)
        sim = sum(tfidf_matrix_list) / len(tfidf_matrix_list)
        sims.append(sim)
    max_sim = max(sims)
    if instruct_type == "Instructions":
        if "train" in args.instruct:
            if max_sim >= 0.90:
                new_items = []
                chosen_instruct = instructions[str(sims.index(max_sim) + 1)]
                print("We can use the LoRA which trained in the order: ", sims.index(max_sim) + 1)
                file_path = args.codebook + "instruction_" + str(sims.index(max_sim) + 1) + ".txt"
                for i in range(len(cur_instruct)):
                    if cur_instruct[i] not in chosen_instruct:
                        new_items.append(cur_instruct[i])
                with open(file_path, 'a', encoding='utf-8') as file:
                    for item in new_items:
                        file.write(f"{item}\n")
            else:
                print("No previous instructions are similar with the current ones, we need to train a new set of LoRA.")
                file_path = args.codebook + "instruction_" + str(len(sims) + 1) + ".txt"
                with open(file_path, 'a', encoding='utf-8') as file:
                    for item in cur_instruct:
                        file.write(f"{item}\n")
        else:
            if max_sim >= 0.90:
                new_items = []
                chosen_instruct = instructions[str(sims.index(max_sim) + 1)]
                print("We can use the LoRA which trained in the order: ", sims.index(max_sim) + 1)
            else:
                print("We can use the LoRA which trained in the order: ", len(sims))
    else:
        if "train" in args.instruct:
            if max_sim >= 0.30:
                new_items = []
                chosen_instruct = instructions[str(sims.index(max_sim) + 1)]
                print("We can use the LoRA which trained in the order: ", sims.index(max_sim) + 1)
                file_path = args.codebook + "instruction_" + str(sims.index(max_sim) + 1) + ".txt"
                for i in range(len(cur_instruct)):
                    if cur_instruct[i] not in chosen_instruct:
                        new_items.append(cur_instruct[i])
                with open(file_path, 'a', encoding='utf-8') as file:
                    for item in new_items:
                        file.write(f"{item}\n")
            else:
                print("No previous instructions are similar with the current ones, we need to train a new set of LoRA.")
                file_path = args.codebook + "instruction_" + str(len(sims) + 1) + ".txt"
                with open(file_path, 'a', encoding='utf-8') as file:
                    for item in cur_instruct:
                        file.write(f"{item}\n")
        else:
            if max_sim >= 0.30:
                new_items = []
                chosen_instruct = instructions[str(sims.index(max_sim) + 1)]
                print("We can use the LoRA which trained in the order: ", sims.index(max_sim) + 1)
            else:
                print("We can use the LoRA which trained in the order: ", len(sims))
    
def main():
    parser = argparse.ArgumentParser(description="Settings of Instruction-Specific LoRA Selection")
    parser.add_argument("--codebook", default="./instruct/codebooks/", type=str, required=False, help="Instruction Dictionary Path")
    parser.add_argument("--instruct", default="./playground/Instructions/ScienceQA/train.json", type=str, required=False, help="Current Instruction Path")
    
    args = parser.parse_args()
    select_lora(args)

    
if __name__ == "__main__":
    main()
