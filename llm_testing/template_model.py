import json
#import requests
from tqdm import tqdm
import sys
import ollama
import numpy


all_lf_questions = {
        'Anti' : [
            ["Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le contraire du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est l'opposé du mot \"", "\"? Donne un seul mot sans ponctuation."]
                ],
        "S_0": [
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun du verbe ou l'adjectif \"","\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun formé à partir du mot \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun formé à partir du verbe ou de l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun dérivé du mot \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Transforme le mot \"", "\" en nom commun. Donne un seul nom commun sans ponctuation."]

        ],
        "Syn_⊂" : [
            ["Quel est le synomyme avec un sens plus large du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Donne un seul mot qui englobe le sens du mot \"", "\"."],
            ["Donne un mot plus général pour signifier \"", "\"."]
        ],
        "A_0" : [
            ["Quel est l'adjectif correspondant au mot \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Transforme le mot \"", "\" en adjectif. Donne un seul adjectif conjugué au masculin, et sans ponctuation."],
            ["Quel est l'adjectif formé à partir du mot \"", "\"? Donne un seul adjectif conjugué au masculin, et sans ponctuation."],
            ["Quel est l'adjectif dérivé du mot \"", "\"? Donne un seul adjectif conjugué au masculin, et sans ponctuation."]
        ],
        "A_2Perf" : [
            ["Quel est l'adjectif correspondant à l'aboutissement de \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Quel est l'adjectif correspondant à la fin de \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Quand on aboutit la chose suivante, quel est l'adjectif approprié pour le définir: :  \"", "\"? Donne un seul adjectif sans ponctuation."]
        ],
        "V_0" : [
            ["Quel est le verbe correspondant au mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Quel est le verbe formé à partir du mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Quel est le verbe dérivé du mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Transforme le mot \"", "\" en verbe. Donne un seul verbe sans ponctuation."]
        ],
        "Syn_⊃^sex" : [
            ["Quel est le mot féminin correspondant au mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le correspondant féminin du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est l'équivalent féminin du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Conjugue le mot \"", "\" au féminin. Donne un seul mot sans ponctuation."],
        ],
        "Adv_0" : [
            ["Quel est l'adverbe correspondant au mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Quel est l'adverbe formé à partir du mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Quel est l'adverbe dérivé du mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Transforme le mot \"", "\" en adverbe. Donne un seul adverbe sans ponctuation."]
        ],
        "S_instr" : [
            ["Quel est l'instrument typiquement utilisé pour faire l'action liée au mot \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Donne le circonstant intrumental typique de \"", "\". Donne un seul nom commun sans ponctuation."],
            ["Quel instrument utilise-t-on pour faire l'acte lié au mot \"", "\"? Donne un seul nom sans ponctuation"]
        ],
        # "Magn" : [
        #     ["Quel est le mot utilisé avec le mot \"", "\" qui amplifie son sens? Donne un seul mot sans ponctuation."],
        #     ["Quel est le mot utilisé avec le mot \"", "\" qui intesifie son sens? Donne un seul mot sans ponctuation."],
        #     ["Donne un mot qui modifie le sens de  \"", "\" en l'amplifiant. Donne un seul mot sans ponctuation."],
        #     ["Je veux amplifier le sens de \"", "\". Quel mot puis-je utiliser avec ce mot pour obtenir un sens amplifié?  Donne un seul mot sans ponctuation."]
        # ],
        "Redun" : [
            ["Donne un mot qui est utilisé comme modificateur redondant du mot \"", "\". Donne un seul mot sans ponctuation."],
            ["Donne un mot qui est utilisé comme modificateur redondant lorsqu'il est ajouté au mot \"", "\". Donne un seul mot sans ponctuation."],
            ["Donne un mot dont le sens est inclus dans celui du mot \"", "\". Donne un seul mot sans ponctuation."]
        ],
        "S_loc": [
            ["Quel est un nom qui décrit la localisation de \"", "\"? Donne un seul nom sans ponctuation."],
            ["Donne le lieu typique de \"", "\"? Donne un seul nom sans ponctuation."],
            ["Donne le lieu ou le moment typique de \"", "\"? Donne un seul nom sans ponctuation."],
            ["À quel endroit se trouve \"", "\"? Donne un seul nom sans ponctuation."]
        ],
        "Cap" : [
            ["Comment appelle-t-on le chef d'un/une \"", "\"? Donne un seul nom sans ponctuation"],
            ["Quel est le terme pour designer le chef d'un/une \"" "\"? Donne un seul nom sans ponctuation"]
        ],
        # Nuance entre Able et A_0???
        # "Able" : [
        #     ["Quel est l'adjectif associé à quelque chose qui peut ou peut être \"", "\"? Donne un seul adjectif sans ponctuation."],
        #     ["Quel est l'adjectif associé à quelque chose qui est \"", "\"? Donne un seul adjectif sans ponctuation."],
        #     ["Comment qualifie-t-on quelque chose qui peut ou peut être \"", "\"? Donne un seul adjectif sans ponctuation."]
        # ]
        "Gener" : [
            ["Quel est un terme générique pour désigner \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est un terme générique qui englobe le mot \"","\"? Donne un seul mot sans ponctuation."],
            ["Quel est le terme qui englobe le mot  \"", "\"? Donne un seul mot sans ponctuation."],
        ],
        # Figur a < 50 exemples
        # "Figur" : [
        #     ["Quel est un terme figuratif qui complète le syntagme \"", "\" de/du/de l' (tout en préservant son sens)? Donne un seul nom commun sans ponctuation."],
        #     ["Quel est un nom métaphorique qui complète le syntagme \"", "\" de/du/de l' (tout en préservant son sens)? Donne un seul nom commun sans ponctuation."],
        #     ["Quel est un nom qui complète la métaphore \"", "\" de/du/de l' ? Donne un seul nom commun sans ponctuation qui permet de garder le sens."]
        # ]
        "Contr" : [
            ["Quel mot crée un contraste, mais n'est pas l'antonyme de, \"", "\" ? Donne un seul mot sans ponctuation."],
            ["Quel est l'opposé de, mais n'est pas l'antonyme de, \"", "\" ? Donne un seul mot sans ponctuation."],
        ],
        "S_res" : [
            ["Quel est un résultat typique de l'acte associé au mot \"", "\" ? Donne un seul mot sans ponctuation"],
            ["Quel est un résultat de l'acte associé au mot \"", "\" ? Donne un seul mot sans ponctuation"]
        ],
        "Sing" : [
            ["Quel est un mot désignant une unité de \"", "\" ? Donne un mot sans ponctuation"],
            ["Comment appelle-t-on une unité de \"", "\" ? Donne un mot sans ponctuation"],
            ["Comment appelle-t-on une seule partie de \"", "\" ? Donne un mot sans ponctuation"],
        ]
    }

questions_exemples = {
    'Anti' : [
        ["L'antonyme du mot \"", "\" est \"", "\"."]
    ],
    "S_0" : [
        []
    ],
    "Syn_⊂" : [
        []
    ],
    "Syn": [
        []
    ],
    "A_0": [
        []
    ],
    "A_2Perf": [
        []
    ],
    "V_0": [
        []
    ],
    "Syn_⊃^sex": [
        []
    ],
    "Adv_0": [
        []
    ],
    "S_instr" : [
        []
    ],
    "Magn": [
        []
    ],
    "Redun": [
        []
    ],
    "S_loc": [
        []
    ],
    "Able": [

    ],
    "Gener": [

    ],
    "Figur": [

    ],
    "Contr": [

    ]

}

k_exemples = {
    'Anti' : [
        ["habiller","déshabiller"],
        ["construire", "détruire"],
        ["petit", "grand"],
        ["chaud", "froid"],
        ["respect", "irrespect"]
    ],
    "S_0" : [
        ["présenter", "présentation"],
        ["partir", "départ"],
        ["proche", "proximité"],
        ["tomber", "chute"],
        ["Pan!", "coup de feu"]
    ],
    "Syn_⊂" : [
        []
    ],
    "A_0": [
        ["temps", "temporel"],
        ["se nourrir", "nutritionnel"],
        ["rotation", "giratoire"],
        ["rapidement", "rapide"],
        ["comparer", "comparatif"]
    ],
    "A_2Perf": [
        []
    ],
    "V_0": [
        ["présentation", "présenter"],
        ["serment", "jurer"],
        ["puant", "puer"],
        ["erreur", "se tromper"],
        ["impatient", "s'impatienter"]
    ],
    "Syn_⊃^sex": [
        []
    ],
    "Adv_0": [
        ["voir", "visuellement"],
        ["alphabet", "alphabétiquement"],
        ["rapide", "rapidement"],
        ["proche", "près"],
        ["vue", "visuellement"]
    ],
    "S_instr" : [
        ["boire", "verre"],
        ["se doucher", "douche"],
        ["sentir", "nez"],
        ["crime", "arme (du)"],
        ["frire", "poêle"]
    ],
    "Redun": [
        []
    ],
    "S_loc": [
        ["boxe", "ring"],
        ["enfant", "enfance"],
        ["fumer", "espace fumeur"],
        ["fumer", "fumoir"],
        ["sentiment", "coeur"]
    ],
    "Cap": [
        ["royaume", "roi"],
        ["mafia", "parrain"],
        ["navire", "capitaine"],
        ["avion", "commandant (de bord)"],
        ["tribu", "chef (de)"]
    ],
    "Able": [
        ["bavarder", "bavard"],
        ["peur", "peureux"]
    ],
    "Gener": [
        ["amour", "sentiment"],
        ["gaz", "substance"],
        ["voir", "percevoir"]
        # todo: 2 autres
    ],
    # "Figur": [
    #     ["pouvoir", "rênes"]
    # ],
    "Contr": [
        ["manger", "boire"],
        ["terre", "mer"],
        ["nous", "eux"],
        ["ici", "là"],
        ["eau", "feu"]
    ],
    # "Equip": [
    #
    # ],
    "S_res": [
        ["soigner", "santé"],
        ["nuire", "dommage"],
        ["assassiner/assassinat", "mort"],
        ["réflexion", "fruit"]
        # todo one more
    ],
    # "Qual": [
    #
    # ],
    "Sing":[
        ["riz", "grain (de)"],
        ["pluie", "goutte (de)"],
        ["battre", "frapper"],
        ["pelage", "poil"],
        ["vocabulaire", "mot"]
    ],
}



    # certaines fonctions lexicales sont difficiles à comprendre, on pourra demander plus de précisions au prof de linguistique.
    # S_1, S_3, S_2^prototyp
    # Conv
    # Real_i


######################### Ask questions individually ################################

# SOURCE: https://github.com/ollama/ollama/blob/main/examples/python-simplegenerate/client.py

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = 'llama3.2' 
context = []

# global values for all samples file & n
file_name, example_file, example_lines, n = '', '', '', 0
fl_ranking = {}

# file_name = './sample_sets/all_relations_50_ex_0.tsv'
#
# examples_file = open(file_name, 'r', encoding="utf-8")
# example_lines = []
#
# for line in examples_file:
#     l = line.replace('\n', '').split('\t')
#     example_lines.append(l)

# Get examples for a given relation from the chosen example file
def get_relation_examples(rel_name):
    global example_lines, file_name, n

    try:
        starting_index = next(i for i, t in enumerate(example_lines) if t == ['>>>', rel_name])
    except StopIteration:
        print(f"Relation '{rel_name}' not in examples.")
        return []

    examples = []
    for i in range(starting_index + 1, min(starting_index + n + 1, len(example_lines))):
        if example_lines[i]:
            examples.append(example_lines[i])
    return examples

# Génère la réponse du LLM
# def generate(prompt, context, file, source_word, expected):
#     print(prompt)
#
#     r = requests.post('http://localhost:11434/api/generate',
#                       json={
#                           'model': model,
#                           'prompt': "Tu es un modèle en français, ne me donne que des réponses d'un mot. " + prompt,
#                           'context': context,
#                       },
#                       stream=True)
#     r.raise_for_status()
#
#     # file structure: source : expected : llm result
#     file.write(f"{source_word} : {expected} : ")
#
#     for i in r.iter_lines():
#         body = json.loads(i)
#         response_part = body.get('response', '')
#
#         # the response streams one token at a time, print that as we receive it
#         #print(response_part, end='', flush=True)
#         file.write(response_part)
#
#         if 'error' in body:
#             raise Exception(body['error'])
#
#         if body.get('done', False):
#             file.write('\n')
#             return body['context']

# Construit la question et le fichier de sortie
# def process_question(relation_type: str, part1:str, part2:str, qIndex : int):
#     # the context stores a conversation history, you can use this to make the model more context aware
#     global context
#
#     # Open a file to store the outputs
#     fileName = f"outputs/{relation_type}_{qIndex}_out.csv"
#
#     # Generate the prompt list to then give as arg to generate()
#     examples = get_relation_examples(relation_type)
#
#     # Clear file contents
#     open("./"+fileName, 'w', encoding="utf-8").close()
#
#     for e in tqdm(examples, desc="Processing examples"):
#         source = e[0]
#         expected = (" ").join(e[1:])
#         question = part1 + source + part2
#         context = generate(question, context, open(fileName, 'a', encoding="utf-8"), source, expected)
#         sys.stdout.flush()
#
#     return fileName

# calcule le success rate une fois que le fichier de sortie est complet
def success_rate(filename):
    file = open('./'+filename, 'r', encoding="utf-8")
    lines = []

    for line in file:
        l = line.replace('\n', '').split(':')
        lines.append(l)

    nb_success = 0
    total = 0

    results = []

    for line in lines:
        if len(line)==3:
            source = str.lower(line[0])
            expected = str.lower(line[1])
            answer = str.lower(line[2]).strip()

            total+=1

            if answer in expected:
                nb_success += 1
                results.append(True)
            else:
                results.append(False)


    success_rate = nb_success/total
    return success_rate

# lf sera en fait une liste de questions pour chaque FL, afin de tester les différents types de verbalisation
# def run_model_for(lf: str, lf_questions : list):
#     # open the SR file (empty)
#     sr_filename = f"./results/sr_{lf}"
#     open(sr_filename, 'w', encoding="utf-8").close()
#     sr_file = open(sr_filename, 'a', encoding="utf-8")
#
#     # q est séparée en 2 parties, "quel est xxx" et "donne xxx"
#     for q in lf_questions:
#         # for each question, run the model & add SR to the SR file
#         fl_file = process_question(lf, q[0], q[1], lf_questions.index(q))
#         sr_file.write(
#             str(lf_questions.index(q)) + " " +
#             str(q[0]) + str(q[1]) + ", sr = " +
#             str(success_rate(fl_file)) + "\n"
#         )


#################### Créer un modèle et le rouler pour des mots ###########################

# Créer le modèle. Non utile apres l'avoir cree une premiere fois
def create_model(question, relation):
    filename = relation + "_model.txt"

    open(filename, 'w', encoding="utf-8").close()
    model_file = open(filename, 'a', encoding="utf-8")
    text_template = " FROM llama3 \n \n SYSTEM \"\"\" \n "+ question +"\n \"\"\" "
    model_file.write(text_template)
    model_file.close()

    ollama.create(model=relation, modelfile=text_template)


# Rouler le modele. k_shot = number of examples we want to add to the question 
def run_model(relation, k_shot):
    questions = all_lf_questions[relation]
    example_sentence = questions_exemples[relation]
    examples = get_relation_examples(relation)
    scores_sublist = []

    # Clear score file
    open('./scores/' + relation, 'a', encoding="utf-8").close()
    score_file = open('./scores/'+relation, 'a', encoding="utf-8")

    for i in range (len(questions)):
        question = questions[i]

        # Open a file to store the outputs
        fileName = f"outputs/{relation}-{i}_k{k_shot}_out.csv"

        # Clear file contents
        open("./"+fileName, 'w', encoding="utf-8").close()

        output_file = open("./"+fileName, 'a', encoding="utf-8")

        for w in tqdm(examples, desc="Processing examples"):
            source = w[0]
            if (k_shot):
                shots = ("Voici " + str(k_shot) + " exemples: ")
                for j in range (k_shot):
                    #shots += (example_sentence[0][0]+k_exemples[relation][j][0]+example_sentence[0][1]+k_exemples[relation][j][1]+example_sentence[0][2]+"\n")

                    shots += ("Pour \""+k_exemples[relation][j][0]+"\", la réponse est \""+k_exemples[relation][j][1]+". \n")
                complete_question =  (str(question[0]) + source + str(question[1]) + "\n" + shots)
                print(complete_question)
            else :
                complete_question =  (str(question[0]) + source + str(question[1]))
            response = ollama.chat(model='relation_general_model', messages=[
                {
                'role': 'user',
                'content': complete_question,
                },
            ])

            expected = (" ").join(w[1:])

            # file structure: source : expected : llm result
            output_file.write(f"{source} : {expected} : {response['message']['content']}\n")

            #print(complete_question)
            #print((f"{source} : {expected} -> {response['message']['content']}"))

        output_file.close()

        score = success_rate(fileName)
        scores_sublist.append(score)

        #score_file.write(f"{question[0]} : {score}\n")

    #score_file.close()
    return scores_sublist

# Mettre juste les sources (pour le moment) des exemples dans un txt ou les valeurs sont separees par des virgules
# Pour les donner à chainforge
# def examples_in_list(examples):
#     f = open("examples.txt", "a")
#     liste = []
#     for w in examples:
#         f.write(w[0] + ',')
#         liste.append(w[0])
#     f.close()
#     return liste

# combiner les prompts en un gros prompt pour simuler le traitement en batch et eviter de re-prompt a chaque fois
# def combined_prompts(question, words):
#     global context
#
#     questions = ""
#     questions_list = []
#     #print(words)
#     for w in words:
#         questions += (str(question[0]) + w[0] + str(question[1])) + '\n'
#         questions_list.append((str(question[0]) + w[0] + str(question[1])))
#
#     # Open a file to store the outputs
#     fileName = 'test.txt'
#
#     # Generate the prompt list to then give as arg to generate()
#     # Clear file contents
#     open("./" + fileName, 'w', encoding="utf-8").close()
#
#     print(questions)
#
#     #context = generate(questions, context, open(fileName, 'a', encoding="utf-8"), '', '')
#
#     sys.stdout.flush()
#
#     return questions_list

# Process all the samples for a given relation
def process_samples(relation, sample_size, num_of_samples, k_shot):
    # Use global values bc filename etc will be used later to get examples
    global file_name, example_file, example_lines, n, fl_ranking

    scores_list = []
    chosen_relation = relation
    n = sample_size
    num_of_samples = num_of_samples

    # Clear score file
    open('./scores/' + chosen_relation, 'w', encoding="utf-8").close()
    score_file = open('./scores/' + chosen_relation, 'a', encoding="utf-8")
    avg_scores = []

    # Iterate through each random sample
    for i in range(num_of_samples):
        file_name = f'./sample_sets/all_relations_{str(n)}_ex_{i}.tsv'
        examples_file = open(file_name, 'r', encoding="utf-8")
        example_lines = []

        for line in examples_file:
            l = line.replace('\n', '').split('\t')
            example_lines.append(l)

        # calc the score for the sample for each question & add to the big list
        scores_for_sample = run_model(chosen_relation, k_shot)
        scores_list.append(scores_for_sample)

    # once all samples are done being evaluated, average all of them & write to file
    for i in range(len(scores_list[0])):
        somme = 0
        question = all_lf_questions[chosen_relation][i]
        for j in range(num_of_samples):
            somme += scores_list[j][i]
        avg_scores.append((round(somme / num_of_samples, 2)))
        score_file.write(question[0] + 'x' + question[1] + ' | ' + str((round(somme / num_of_samples), 2)) + '\n')
    # Add best score to the FL ranking along w its question
    maximum = max(avg_scores)
    fl_ranking[chosen_relation] = [maximum, all_lf_questions[chosen_relation][avg_scores.index(maximum)]]

def main():

    # for k in [0,1,3,5]:
    #     process_samples('Anti', 30, 2, k)

    #process_samples('Anti', 30, 2, 0)

    for rel in all_lf_questions.keys():
        print(rel)
        process_samples(rel, 30, 2, 0)

    # todo: à ameliorer (affichage, lecture du map)
    # open('./scores/fl_ranking', 'w', encoding="utf-8").close()
    # ranking_file = open('./scores/fl_ranking', 'a', encoding="utf-8")
    # fl_ranking_sorted = dict(sorted(fl_ranking.items(), key=lambda item: item[1], reverse=True))
    # for fl in fl_ranking_sorted.keys():
    #     ranking_file.write(str(fl) + '\n' + str(fl_ranking_sorted[fl][0]) + '|' + str(fl_ranking_sorted[fl][1][0]) + '\n')


if __name__ == "__main__":
    main()