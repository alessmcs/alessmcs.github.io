# Authors: Alexander Petrov (original API),
#           Viviane Binet, Alessandra Mancas

import math
import random
import re
import pandas as pd
import xml.etree.ElementTree as ET

from PIL.TiffImagePlugin import IFDRational
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import pickle
import numpy as np
import csv

# Path aless
LS_FR_PATH = "../lexical-system-fr/10/ls-fr-V3.1/"
# Path vivi
# LS_FR_PATH = "C:/_Code/courses/IFT3150_projet/lexical-system-fr/7/ls-fr-V2.1/"

# Extract center word from a RL-fr entry
def extract_juice(raw):
    pattern = r"<span class='namingform'>(.*?)</span>"
    s = re.search(pattern, raw).group(1)
    s = s.replace('\xa0', '')       # this '\xa0' thing is some sort of non-breaking space character, its pesky, so we remove it
    return s

# Creates dataframe object for desired number.
# For example, if i == 2, we build dataframe2, where the 2 comes from the resource
# We can always just paste the big string in a main function, and go from there
def get_df(i: int):
    dfs = r"""
    df1 = pd.read_csv(LS_FR_PATH+"01-lsnodes.csv", sep='\t')
    df2 = pd.read_csv(LS_FR_PATH+"02-lsentries.csv", sep='\t')
    df4 = pd.read_csv(LS_FR_PATH+"04-lscopolysemy-rel.csv", sep='\t')
    df6 = pd.read_csv(LS_FR_PATH+"06-lsgramcharac-rel.csv", sep='\t')
    df8 = pd.read_csv(LS_FR_PATH+"08-lswordforms.csv", sep='\t')
    df10 = pd.read_csv(LS_FR_PATH+"10-lssemlabel-rel.csv", sep='\t')
    df11 = pd.read_csv(LS_FR_PATH+"11-lspropform-rel.csv", sep='\t')
    df13 = pd.read_csv(LS_FR_PATH+"13-lsdef.csv", sep='\t')
    df15 = pd.read_csv(LS_FR_PATH+"15-lslf-rel.csv", sep='\t')
    df17 = pd.read_csv(LS_FR_PATH+"17-lsex.csv", sep='\t')
    df18 = pd.read_csv(LS_FR_PATH+"18-lsex-rel.csv", sep='\t')
    """

    pattern = "df([0-9]+) = (.*?)\n"
    harvests = re.findall(pattern, dfs)

    for pair in harvests:
        n, s = pair
        if str(i) == n:
            df = eval(s)
            df = df.fillna(0)  # This removes those pesky NaN stuff and makes them render as actual empty strings ''

            return df

    print("none")

# Get a word from its mapped content
def get_word_from_id(id: str, mapping) -> str:
    return mapping[id]

# Get the entire word map from df1
def get_word_from_id_mapping():
    df1 = get_df(1)
    mapping = {}

    ids = []
    words = []

    n = len(df1)
    for i in range(n):
        word = extract_juice(df1["lexname"].iloc[i])
        words.append(word)

        ids.append(df1["id"].iloc[i])

    for word, id in zip(words, ids):
        mapping[id] = word
    return mapping

# Get relation endpoints for each example in df15
def get_endpoints(df15, i: int, mapping, name_rather_than_id=True) -> (str, str):

    source_id = df15.iloc[i]['source']
    target_id = df15.iloc[i]['target']

    if name_rather_than_id:
        source = mapping[source_id]
        target = mapping[target_id]
    else:
        source = source_id
        target = target_id

    return source, target

# Returns a dictionary with the relation ids, to save time when we have to search
def get_relation_dict():
    # Parse the XML file
    tree =  ET.parse(LS_FR_PATH+"14-lslf-model.xml")

    root = tree.getroot()

    relations = {}

    # Iterate over <family> elements
    for family in root.findall('.//family'):
        # Initialize an empty dictionary for the current family
        family_id = family.attrib['id']
        family_name = family.attrib['name']

        # Iterate over <lexicalfunction> elements within the current <family>
        for lexicalfunction in family.findall('.//lexicalfunction'):
            lf_id = lexicalfunction.attrib['id']
            lf_name = lexicalfunction.attrib['name']
            lf_linktype = lexicalfunction.attrib["linktype"]

            # Create a dictionary for the current lexical function
            relations[lf_id] = (lf_name, family_name, lf_linktype)

    return relations

# Get the name of a relation from its ID
def get_relation_name(df, i: int, relations) -> (str, str):

    relation_id = df.iloc[i]['lf']
    separator = df.iloc[i]['separator']

    name, family, linktype = relations[relation_id]
    return name, family, separator

# todo: delete???
def clean_dfs(df15, df2, no_numbers=True, no_locutions=False):
    df2_copy = df2.copy()
    modified_ids = df2.copy()['id'].apply(lambda word: word.replace('entry', 'node'))
    df2_copy['modified_id'] = modified_ids

    df15_copy = df15.copy()

    # Define a boolean function to do filtering on dfs
    condition = ''
    if no_numbers and not no_locutions:
        condition = lambda word : not (any(char.isdigit() for char in word))
    elif not no_numbers and no_locutions:
        condition = lambda word : False if len(word.split(' '))>1 else True
    elif no_numbers and no_locutions:
        condition = lambda word : False if (any(char.isdigit()) for char in word) or len(word.split(' '))>1 else True

    df2_cleaned = df2_copy[df2_copy['name'].apply(condition)]
    ok_indexes = df2_cleaned['modified_id']
    print(ok_indexes)

    print(df15_copy)

    df15_cleaned_1 = df15_copy[df15_copy['source'].isin(ok_indexes)]
    print(df15_cleaned_1)
    df15_cleaned_2 = df15_cleaned_1[df15_cleaned_1['target'].isin(ok_indexes)]

    return df2_cleaned, df15_cleaned_2

# Map - keys => FL (relation), values => map where key = source & value = examples
def create_map_to_write(name_rather_than_id=True):
    df15 = get_df(15)
    df2 = get_df(2)

    relations = get_relation_dict()
    mapping = get_word_from_id_mapping()

    # Remove numbers and/or locutions to simplify examples and control variance
    df2_cleaned, df15_cleaned = clean_dfs(df15, df2)

    N = len(df15_cleaned)

    everything = {}

    # Pour chaque mot dans les listes
    for i in tqdm(range(N), desc="creating map", ascii=True):
        source, target = get_endpoints(df15_cleaned, i, mapping, name_rather_than_id)
        relation, family, separator = get_relation_name(df15_cleaned, i, relations)

        # enlever tous les entries qui contiennent des chiffres
        condition = lambda word : True if any (char.isdigit() for char in word) else False

        if (any (char.isdigit() for char in source)) or (any (char.isdigit() for char in target)):
            continue

        # Append to everything with or without the separator
        if relation not in everything:
            everything[relation] = {}
            everything[relation][source] = []
            #everything[relation][source].append([target, family])
            everything[relation][source].append([target, family, separator])

        else:
            if source not in everything[relation]:
                everything[relation][source] = []
                #everything[relation][source].append([target, family])
                everything[relation][source].append([target, family, separator])
            else:
                #everything[relation][source].append([target, family])
                everything[relation][source].append([target, family, separator])

    # print(everything['Hypo']['appartement'])

    return everything

# Get a sample from everything depending on the min number of examples of each relation
# (to make sure we have enough examples to work with)
def create_map_most_examples(everything:dict, min_number:int, nb_examples:int):
    small_map = {}
    for relation in everything:
        count = len(everything[relation])

        if count>= min_number:
            small_map[relation]={}
            key_sample = random.sample(sorted(everything[relation]),nb_examples)
            dict_sample = {k: everything[relation][k] for k in key_sample}
            small_map[relation] = dict_sample

    #print(small_map)
    return small_map


### FILE WRITING METHODS ###

# Put every set of source-targets in a big txt file
def write_readable_file(everything: dict, filename: str):
    with open(filename, 'w', encoding="utf-8") as f:
        for relation in everything:
            f.write('\n' + relation + '\n')
            for exemple_key in everything[relation]:
                string = str(exemple_key) + ' => '
                # Pour chaque elem du tableau de la clé, écrire juste le 1er elem + separateur
                for target in everything[relation][exemple_key]:
                    string += str(target[2]) + str(target[0]) + ' '
                string += '\n'
                f.write(string)

# Fichier tab separated values car il y a déjà les séparateurs ; et ,
# Format des colonnes:
# mot_input     mot_output_1    mot_output_2    mot_output_3    ...
def write_tsv(everything:dict, filename:str):
    with open(filename, 'w', encoding="utf-8") as f:
        for relation in everything:
            f.write('>>>\t'+ relation + '\n')
            for exemple_key in everything[relation]:
                string = str(exemple_key) + '\t'
                # Pour chaque elem du tableau de la clé, écrire juste le 1er elem + separateur
                for target in everything[relation][exemple_key]:
                    string += str(target[2]) + str(target[0]) + '\t'
                string += '\n'
                f.write(string)

def read_tsv(file_to_read):
    dictionary = {}
    with open(file_to_read, encoding="utf-8") as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            if line[0]== ">>>":     # nouvelle relation
                relation = line[1]
                dictionary[relation] = {}
            else:
                source = line[0]
                dictionary[relation][source] = line[1:]
    return dictionary

# Create x random sample sets of size y for each relation with a min number of examples
# Put everything in a big tsv file so we can parse it in template_model
def create_sample_sets(min_number, nb_examples, nb_sets, all_results):
    print(len(all_results))

    for i in range(nb_sets):
        curr_set = create_map_most_examples(all_results, min_number, nb_examples)
        #print(curr_set)
        write_tsv(curr_set, f"./llm_testing/sample_sets/all_relations_{nb_examples}_ex_{i}.tsv")

if __name__ == "__main__":

    # Get all the relations & examples
    everything = create_map_to_write()
    # Create as many samples of desired size as I want
    create_sample_sets(30, 30, 3, everything)

    # todo: pertinence des parties du discours ou pas??



