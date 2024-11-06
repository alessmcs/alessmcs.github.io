# Master Thesis
# Author: Alexander Petrov
#from asyncio import write

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

# Path vivi
# LS_FR_PATH = "C:/_Code/courses/IFT3150_projet/lexical-system-fr/7/ls-fr-V2.1/"

# Path aless
LS_FR_PATH = "../lexical-system-fr/10/ls-fr-V3.1/"



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

# get word /from/ id
def get_word_form_id(id: str, mapping) -> str:


    return mapping[id]

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

    # for each word in the word mapping, get its id & add subscript & superscript


def get_endpoints(df15, df2, i: int, mapping, sub_sup, name_rather_than_id=True) -> (str, str):

    # pdd_ids = df2['id'].to_numpy()
    # ids = []
    # for i in range(len(pdd_ids)):
    #     num_id = pdd_ids[i].replace('entry', 'node')
    #     ids.append(num_id)

    source_id = df15.iloc[i]['source']
    target_id = df15.iloc[i]['target']

    # source_subscript, source_superscript = "", ""
    # target_subscript, target_superscript = "", ""
    #
    # # Check if source_id exists in df2
    # if source_id in ids:
    #     source_subscript = '_' + str(df2.loc[df2['id'] == source_id.replace('node', 'entry'), 'subscript'].values)
    #     source_superscript = '^' + str(df2.loc[df2['id'] == source_id.replace('node', 'entry'), 'superscript'].values)
    #
    # # Check if target_id exists in df2
    # if target_id in ids:
    #     target_subscript = '_' + str(df2.loc[df2['id'] == target_id, 'subscript'].values)
    #     target_superscript = '^' + str(df2.loc[df2['id'] == target_id, 'superscript'].values)

    if name_rather_than_id:
        # if sub_sup :
        #     source = mapping[source_id] + source_subscript + source_superscript
        #     target = mapping[target_id] + target_subscript + target_superscript
        # else:
        source = mapping[source_id]
        target = mapping[target_id]
    else:
        source = source_id
        target = target_id

    return source, target


def get_merged_values(df15, i: int) -> int:
    # df = get_df(15)
    merged_value = df15.iloc[i]["merged"]
    return merged_value


# -- POUR PROJET --
# Returns a dictionary with the relation ids, which takes a lot less time when we have to search
def get_relation_dict():
    # Parse the XML file
    # tree = ET.parse("lexical-system-fr/9/ls-fr-V3/14-lslf-model.xml")
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


def get_relation_name(df15, i: int, relations) -> (str, str):
    # df15 = get_df(15)

    relation_id = df15.iloc[i]['lf']
    separator = df15.iloc[i]['separator']

    name, family, linktype = relations[relation_id]
    return name, family, separator


def create_map_to_write(name_rather_than_id=True):
    df15 = get_df(15)
    df2 = get_df(2)

    N = len(df15)

    relations = get_relation_dict()
    mapping = get_word_from_id_mapping()

    everything = {}

    for i in tqdm(range(N), desc="creating map", ascii=True):
        source, target = get_endpoints(df15, df2, i, mapping, True, name_rather_than_id)
        relation, family, separator = get_relation_name(df15, i, relations)

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


    #for fl in everything:
    #    print(str(fl) + " " + str(everything[fl]))

    return everything

def create_map_most_examples(everything:dict, min_number:int, nb_examples:int):
    small_map = {}
    for relation in everything:
        count = len(everything[relation])

        if count>= min_number:
            small_map[relation]={}
            key_sample = random.sample(sorted(everything[relation]),nb_examples)
            dict_sample = {k: everything[relation][k] for k in key_sample}
            small_map[relation] = dict_sample

    print(small_map)
    return small_map



def get_names():
    df1 = get_df(1)

    names = []
    for i in range(len(df1)):
        names.append(extract_juice(df1['lexname'][i]))

    unique_names = list(set(names))

    return names, unique_names


def get_freq_dict(names):

    # Count the frequency of the number of words in each name
    freq_dict = Counter(len(name.split()) for name in names)

    print(freq_dict)

    num_words = list(freq_dict.keys())
    frequency = list(freq_dict.values())

    # Plotting the bar chart
    plt.bar(num_words, frequency, color='skyblue')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Frequency of Number of Words in Names')
    plt.show()

    return freq_dict


# for doing the synonyms file (begin)

# plan:
# I have a source word, and for a given (sub-)relation, I want to find all target words related to that source
# To this end, we want to go through the (sub)-relation wherein we have a pair with the source word as source,
# and extract its target value
# Since the resource is labeled by id, i need to find
def get_targets(source_id, relation_id):
    df15 = get_df(15)

    targets = []

    # Regarder seulement les rows de df15 qui correspondent à la source & la FL
    filtered_df = df15[(df15["source"] == source_id) & (df15["lf"] == relation_id)]

    targets = list(
        filtered_df[["target", "form", "separator", "merged", "syntacticframe", "constraint", "position"]]
        .itertuples(index=False, name=None)
    )

    # for i in range(len(df15)):

    #     if df15.iloc[i]["source"] == source_id and df15.iloc[i]["lf"] == relation_id:
    #         target_plus_merged_pair = (
    #             df15.iloc[i]["target"], df15.iloc[i]["form"], df15.iloc[i]["separator"], df15.iloc[i]["merged"],
    #             df15.iloc[i]["syntacticframe"], df15.iloc[i]["constraint"], df15.iloc[i]["position"])
    #         targets.append(target_plus_merged_pair)
    return targets


def get_row_info():
    df15 = get_df(15)

    n = len(df15)

    neighbours = {}

    for i in tqdm(range(n)):

        source_id = df15.iloc[i]["source"]
        relation_id = df15.iloc[i]["lf"]
        target_id = df15.iloc[i]["target"]
        # form = df15.iloc[i]["form"]
        # separator = df15.iloc[i]["separator"]
        # merged = df15.iloc[i]["merged"]
        # syntactic_frame = df15.iloc[i]["syntacticframe"]
        # constraint = df15.iloc[i]["constraint"]
        # position = df15.iloc[i]["position"]

        if relation_id not in neighbours:
            neighbours[relation_id] = []

        neighbours[relation_id].append((source_id, get_targets(source_id, relation_id), relation_id))

    return neighbours


def translate_lists(alist):
    translated_lists = []
    for e in alist:
        target_id, form_flag, sep_flag, merge_flag, syntax_flag, constraint_flag, position_flag = e
        word = get_word_form_id(target_id)
        translated_lists.append(
            (word, form_flag, sep_flag, merge_flag, syntax_flag, constraint_flag, position_flag))
    return translated_lists

def translate_ids(my_dict):
    translated_dict = {}
    for key in tqdm(my_dict):

        # this line is to translate the relation from id into a name
        name, family = relations[key]

        for element in my_dict[key]:
            first, second = element

            if name not in translated_dict:
                translated_dict[name] = []

            if (get_word_form_id(first), translate_lists(second)) not in translated_dict[name]:
                translated_dict[name].append((get_word_form_id(first), translate_lists(second)))

    return translated_dict


###


def revive_dict(s):

    pattern = r'np\.(int64|float64)\(([\d\.]+)\)'

    # Function to replace matches with the captured number
    def replace_np_number(match):
        return match.group(2)  # Return the captured number inside ()

    # Perform the replacement using re.sub() with the defined pattern and replacement function
    output_string = re.sub(pattern, replace_np_number, s)


    my_dict = eval(output_string)
    return my_dict


def remove_duplicates(my_dict):
    def tuple_to_hashable(tpl):
        """ Convert tuple to a hashable form where lists are converted to tuples recursively """
        if isinstance(tpl, list):
            return tuple(tuple_to_hashable(item) if isinstance(item, (tuple, list)) else item for item in tpl)
        elif isinstance(tpl, tuple):
            return tuple(tuple_to_hashable(item) if isinstance(item, (tuple, list)) else item for item in tpl)
        else:
            return tpl

    def remove_duplicates_from_list(lst):
        seen = set()
        result = []
        for tpl in lst:
            tpl_hashable = tuple_to_hashable(tpl)
            if tpl_hashable not in seen:
                seen.add(tpl_hashable)
                result.append(tpl)
        return result

    for relation in my_dict:


        my_dict[relation] = remove_duplicates_from_list(my_dict[relation])

    return my_dict


def clean_synonyms_dict(my_dict):          # cleans away all the merge-and-other info ?
    new_dict = {}
    for relation in my_dict:
        source_pairs = []
        for source_element in my_dict[relation]:
            source, targets_list = source_element[0], source_element[1]
            targets = []
            for target_list in targets_list:
                target = target_list[0]
                targets.append(target)
            source_pairs.append((source, targets))
        new_dict[relation] = source_pairs
    return new_dict



# for doing the synonyms file (end)

def translate_synonym_ids_via_mapping(name_mapping, relation_mapping):

    '''
    needs these first:

    name_mapping = get_word_from_id_mapping()
    relation_mapping = get_relation_dict()

    :param name_mapping:
    :param relation_mapping:
    :return:
    '''

    trans_synonyms = {}
    for rel in synonyms:
        new_source_pair_list = []
        for source_pair in synonyms[rel]:
            source_id, target_id_list = source_pair
            source_name = name_mapping[source_id]

            target_name_list = []
            for target_id in target_id_list:
                target_name = name_mapping[target_id]
                target_name_list.append(target_name)

            new_source_pair = (source_name, target_name_list)
            new_source_pair_list.append(new_source_pair)
        rel_name = relation_mapping[rel][0]     # we want the name, not the family name, hence the [0]
        trans_synonyms[rel_name] = new_source_pair_list

    trans_synonyms2 = trans_synonyms
    # trans_synonyms2 = remove_duplicates(trans_synonyms)     # else, this leaves you with duplicates, so we call remove_duplicates

    # makes it in a mapping-style of dict
    for rel in trans_synonyms2:
        new_dict = {}
        for source_pair in trans_synonyms2[rel]:
            source_name, target_name_list = source_pair
            new_dict[source_name] = target_name_list

        trans_synonyms2[rel] = new_dict



    return trans_synonyms2


def get_para_synta_relations():
    relations = get_relation_dict()

    paradigmatics = []
    syntagmatics = []

    for key in relations:
        name, family, linktype= relations[key]
        if linktype == "paradigmatic":
            paradigmatics.append(name)
        elif linktype == "syntagmatic":
            syntagmatics.append(name)
        else:
            raise ValueError("Neither paradigmatic nor syntagmatic")
    return paradigmatics, syntagmatics


def write_readable_file(everything:dict, filename:str):
     with open(filename, 'w', encoding="utf-8") as f:
        for relation in everything:
            f.write('\n' +relation + '\n')
            for exemple_key in everything[relation]:
                string = str(exemple_key) + ' => '
                # Pour chaque elem du tableau de la clé, écrire juste le 1er elem + separateur
                for target in everything[relation][exemple_key]:
                    string += str(target[2]) + str(target[0]) + ' '
                    #string += str(target[0])
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

# write as csv pour chainforge???


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

def create_sample_sets(min_number, nb_examples, nb_sets, all_results):
    for i in range(nb_sets):
        curr_set = create_map_most_examples(all_results, min_number, nb_examples)
        print(curr_set)
        write_tsv(curr_set, f"./llm_testing/sample_sets/all_relations_{nb_examples}_ex_{i}.tsv")


if __name__ == "__main__":
    # Get all the relations & examples
    everything = create_map_to_write(True)
    # create n samples
    create_sample_sets(100, 100, 3, everything)

    # TODO: use ids!!

    # todo: sort & format the file?

    # faire une fonction pour print examples given a certain FL??
    # inclure parties du discours????








