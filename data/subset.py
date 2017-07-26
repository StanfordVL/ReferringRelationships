import json
import sys

objects = ["man", "shoes", "jacket", "sunglasses"]
relationships = ["wears", "has", "have"]
triplet_subset = [("man", "wears", "shoes")]


def _in_subset(object, relationship, subject, object_subset=objects, relationship_subset=relationships):
    return object in object_subset and subject in object_subset and relationship in relationship_subset


def in_subset(object, relationship, subject, triplet_subset=triplet_subset):
    return (object, relationship, subject) in triplet_subset


def get_object_name(rel_data, object_type):
    if "name" in rel_data[object_type].keys():
        return rel_data[object_type]["name"]
    else:
        return rel_data[object_type]["names"][0]


if __name__ == "__main__":
    print("loading json file...")
    data = json.load(open("relationships.json"))
    print("json loaded!")
    subset_data = []
    nb_images = len(data)
    for i in range(nb_images):
        if i % 1000 == 0:
            print("processed {} images out of {}".format(i, nb_images))
        added = False
        image_id = data[i]["image_id"]
        relationships = data[i]["relationships"]
        subset_relationships = []
        for j, rel_data in enumerate(relationships):
            object_name = get_object_name(rel_data, "object")
            subject_name = get_object_name(rel_data, "subject")
            relationship_name = rel_data["predicate"]
            if in_subset(object_name, relationship_name, subject_name):
                subset_relationships += [rel_data]
                added = True
        if added:
            subset_data += [{"image_id": image_id,
                             "relationships": subset_relationships}]
    json.dump(subset_data, open("subset_1/subset_relationships.json", "w"))
