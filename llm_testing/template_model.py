import json
#import requests
from tqdm import tqdm
import sys
import ollama
import numpy
import pandas as pd
import matplotlib.pyplot as plt


# Ce sont toutes des fl paradigmatiques
all_lf_questions = {
        ### FLS SIMPLES ###
        'Anti' : [
            ["Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le contraire du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est l'opposé du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel mot a un sens correspondant au sens de \"", "\" dans lequel est insérée une négation? Donne un seul mot sans ponctuation."] # Ressource
                ],
        "S_0": [
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun du verbe ou l'adjectif \"","\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun formé à partir du mot \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun formé à partir du verbe ou de l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun dérivé du mot \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun dont \"", "\" est la racine? Donne un seul nom commun sans ponctuation."],
            ["Transforme le mot \"", "\" en nom commun. Donne un seul nom commun sans ponctuation."],
            ["Quelle est une lexie nominale ayant le même sens que \"", " ? Donne un seul nom commun sans ponctuation."] # Ressource
        ],
        "Syn_⊂" : [
            ["Quel est le synonyme avec un sens plus large du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel mot a un sens plus large  \"", "\"? Donne un seul mot sans ponctuation."],
            ["Donne un seul mot qui englobe le sens du mot \"", "\"."],
            ["Donne un mot plus général pour signifier \"", "\"."]
        ],
        "A_0" : [
            ["Quel est l'adjectif correspondant au mot \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Transforme le mot \"", "\" en adjectif. Donne un seul adjectif conjugué au masculin, et sans ponctuation."],
            ["Quel est l'adjectif formé à partir du mot \"", "\"? Donne un seul adjectif conjugué au masculin, et sans ponctuation."],
            ["Quel est l'adjectif dérivé du mot \"", "\"? Donne un seul adjectif conjugué au masculin, et sans ponctuation."],
            ["Quelle est une lexie adjectivale ayant le même sens (ou presque) que \"", " ? Donne un seul adjectif conjugué au masculin, et sans ponctuation."]
            # Ressource
        ],
        "V_0" : [
            ["Quel est le verbe correspondant au mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Quel est le verbe formé à partir du mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Quel est le verbe dérivé du mot \"", "\"? Donne un seul verbe sans ponctuation."],
            ["Transforme le mot \"", "\" en verbe. Donne un seul verbe sans ponctuation."],
            ["Quelle est une lexie verbale ayant le même sens que \"", " ? Donne un seul verbe sans ponctuation."]
            # Ressource
        ],
        "Adv_0" : [
            ["Quel est l'adverbe correspondant au mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Quel est l'adverbe formé à partir du mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Quel est l'adverbe dérivé du mot \"", "\"? Donne un seul adverbe sans ponctuation."],
            ["Transforme le mot \"", "\" en adverbe. Donne un seul adverbe sans ponctuation."],
            ["Quelle est une lexie adverbiale ayant le même sens que \"", " ? Donne un seul adverbe sans ponctuation."]
            # Ressource
        ],
        "S_loc": [
            ["Quel est un nom qui décrit la localisation de \"", "\"? Donne un seul nom sans ponctuation."],
            ["Donne le lieu typique de \"", "\"? Donne un seul nom sans ponctuation."],
            ["Donne le lieu ou le moment typique de \"", "\"? Donne un seul nom sans ponctuation."],
            ["À quel endroit se trouve \"", "\"? Donne un seul nom sans ponctuation."],
            ["Quelle est une lexie nominale qui désigne un circonstant de lieu typique de, ou d'un fait lié à \"", "\"? Donne un seul nom sans ponctuation"] # Ressource
        ],
        "Gener" : [
            ["Quel est un terme générique pour désigner \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est un terme générique qui englobe le mot \"","\"? Donne un seul mot sans ponctuation."],
            ["Quel est le terme qui englobe le mot  \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est un hyperonyme classifiant de \"", "\" ? Donne un seul mot sans ponctuation."] # Ressource
        ],
        "Contr" : [
            ["Quel mot crée un contraste avec, mais n'est pas l'antonyme de, \"", "\" ? Donne un seul mot sans ponctuation."],
            ["Quel est l'opposé, mais pas l'antonyme de, \"", "\" ? Donne un seul mot sans ponctuation."],
            ["Quel mot est l'opposé de \"", "\", sans être son contraire? Donne un seul mot sans ponctuation."],
            ["Quel mot est en opposition contrastive sur le plan sémantique avec \"", " ? Donne un seul mot sans ponctuation"] # ressource
        ],
        "S_res" : [
            ["Quel est un résultat typique de l'acte associé au mot \"", "\" ? Donne un seul mot sans ponctuation"],
            ["Quel est un résultat de l'acte associé au mot \"", "\" ? Donne un seul mot sans ponctuation"],
            ["En quoi résulte l'acte de \"", "\" ? Donne un seul mot sans ponctuation"],
            ["Quelle est une lexie nominale qui désigne un circonstant de résultat typique de \"",
             "\"? Donne un seul mot sans ponctuation"]  # Ressource
        ],
        "Sing" : [
            ["Quel est un mot désignant une unité de \"", "\" ? Donne un mot sans ponctuation"],
            ["Comment appelle-t-on une unité de \"", "\" ? Donne un mot sans ponctuation"],
            ["Comment appelle-t-on une seule partie de \"", "\" ? Donne un mot sans ponctuation"],
            ["Donne une lexie qui désigne une unité de \"", "\" ? Donne un mot sans ponctuation"] #ressource
        ],

        ### FLS COMPLEXES ###
        "A_2Perf" : [
            ["Quel est l'adjectif correspondant à l'aboutissement de \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Quel est l'adjectif correspondant à la fin de \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Quand on aboutit la chose suivante, quel est l'adjectif approprié pour le définir: :  \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Comment peut-on qualifier la chose suivante lorsqu'elle est aboutie :  \"","\"? Donne un seul adjectif sans ponctuation."],
            ["Comment qualifie-t-on l'actant du mot \"", "\" lorsqu'il finit l'acte associé au mot donné?"],
            ["Quel est le qualificatif adjectival pour désigner l'aboutissement du fait \"", "\"? Donne un adjectif sans ponctuation"] #ressource
        ],
        "Syn_⊃^sex" : [
            ["Quel est le mot féminin correspondant au mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le correspondant féminin du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est l'équivalent féminin du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Conjugue le mot \"", "\" au féminin. Donne un seul mot sans ponctuation."],
            # rien trouvé dans la ressource?
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
        ['triplex', 'appartement'],
        ['tambouriner', 'frapper'],
        ['à la vitesse de [Y]', 'rapidement'],
        ['lune de miel', 'voyage'],
        ['cyclone', 'vent']
    ],
    "A_0": [
        ["temps", "temporel"],
        ["se nourrir", "nutritionnel"],
        ["rotation", "giratoire"],
        ["rapidement", "rapide"],
        ["comparer", "comparatif"]
    ],
    "A_2Perf": [
        ['barrer', 'barré'],
        ['allonger', 'couché'],
        ['adopter', 'adoptif'], #????
        ['voir', 'vu'],
        ['vider', 'vide']
    ],
    "V_0": [
        ["présentation", "présenter"],
        ["serment", "jurer"],
        ["puant", "puer"],
        ["erreur", "se tromper"],
        ["impatient", "s'impatienter"]
    ],
    "Syn_⊃^sex": [
        ['boulanger', 'boulangère'],
        ['cheval', 'jument'],
        ['curieux', 'curieuse'],
        ['prêtre', 'prêtresse'],
        ['égoïste', 'égoïste']
    ],
    "Adv_0": [
        ["voir", "visuellement"],
        ["alphabet", "alphabétiquement"],
        ["rapide", "rapidement"],
        ["proche", "près"],
        ["vue", "visuellement"]
    ],
    "S_loc": [
        ["boxe", "ring"],
        ["enfant", "enfance"],
        ["fumer", "espace fumeur"],
        ["fumer", "fumoir"],
        ["sentiment", "coeur"]
    ],
    "Gener": [
        ["amour", "sentiment"],
        ["gaz", "substance"],
        ['table', 'meuble'],
        ["voir", "percevoir"],
        ['chien', 'animal']
    ],
    "Contr": [
        ["manger", "boire"],
        ["terre", "mer"],
        ["nous", "eux"],
        ["ici", "là"],
        ["eau", "feu"]
    ],
    "S_res": [
        ['se décler', 'décalage'],
        ["soigner", "santé"],
        ["nuire", "dommage"],
        ['examen', 'résultat'],
        ["assassiner/assassinat", "mort"],
    ],
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

# global values for all samples file & n
file_name, example_file, example_lines, n = '', '', '', 0
all_fls_df = pd.DataFrame()
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
    #example_sentence = questions_exemples[relation]
    examples = get_relation_examples(relation)
    scores_sublist = []

    # Clear score file
    open('./scores/' + relation, 'a', encoding="utf-8").close()
    score_file = open('./scores/'+relation, 'a', encoding="utf-8")

    for i in range (len(questions)):
        question = questions[i]
        #print('Question ' + str(i))

        # Open a file to store the outputs
        fileName = f"outputs/{relation}-{i}_k{k_shot}_out.csv"

        # Clear file contents
        open("./"+fileName, 'w', encoding="utf-8").close()

        output_file = open("./"+fileName, 'a', encoding="utf-8")

        for w in tqdm(examples, desc=f"Processing examples question {i}"):
            source = w[0]
            if (k_shot):
                shots = ("Voici " + str(k_shot) + " exemples: ")
                for j in range (k_shot):
                    #shots += (example_sentence[0][0]+k_exemples[relation][j][0]+example_sentence[0][1]+k_exemples[relation][j][1]+example_sentence[0][2]+"\n")

                    shots += ("Pour \""+k_exemples[relation][j][0]+"\", la réponse est \""+k_exemples[relation][j][1]+". \n")
                complete_question =  (str(question[0]) + source + str(question[1]) + "\n" + shots)
                #print(complete_question)
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
def process_samples(relation, sample_size, num_of_samples):
    # Use global values bc filename etc will be used later to get examples
    global file_name, example_file, example_lines, n, fl_ranking, all_fls_df

    chosen_relation = relation
    n = sample_size
    #num_of_samples = num_of_samples

    # Clear score file
    open('./scores/' + chosen_relation, 'w', encoding="utf-8").close()
    score_file = open('./scores/' + chosen_relation, 'a', encoding="utf-8")
    avg_scores = []

    # Create list of dictionaries to transform in a Dataframe after
    list_of_dict = []

    # Iterate through each random sample
    for i in range(num_of_samples):
        scores_list = []
        file_name = f'./sample_sets/all_relations_{str(n)}_ex_{i}.tsv'
        examples_file = open(file_name, 'r', encoding="utf-8")
        # TODO: what does this do?
        example_lines = []
        k_shot = [0,1,3,5]

        for line in examples_file:
            l = line.replace('\n', '').split('\t')
            example_lines.append(l)

        for k in range(len(k_shot)):
            # calc the score for the sample for each question & add to the big list
            print(file_name)
            scores_for_sample = run_model(chosen_relation, k_shot[k])
            scores_list.append(scores_for_sample)

        # once all samples are done being evaluated, average all of them & write to file
            for l in range(len(scores_list[0])):  #nb de questions
                #somme = 0
                #for j in range(num_of_samples):
                    #somme += scores_list[j][l]
                    #avg_scores.append((round(somme / num_of_samples, 2)))

                dict1 = {"relation": relation, "no_question": l, "no_echantillon":i, f"score{k_shot[k]}" : round(scores_list[k][l], 2)}
                list_of_dict.append(dict1)

            #score_file.write(question[0] + 'x' + question[1] + ' | ' + str((round(somme / num_of_samples), 2)) + '\n')
    # Add best score to the FL ranking along w its question
    #maximum = max(avg_scores)
    # fl_ranking[chosen_relation] = [maximum, all_lf_questions[chosen_relation][avg_scores.index(maximum)]]

    df = pd.DataFrame(list_of_dict)
    concatenated_df = pd.concat([all_fls_df, df])
    all_fls_df = concatenated_df
    all_fls_df.to_csv("df_complete.csv")
    print(df)
    return all_fls_df





def create_df_by_relation(relation, df):
    df_relation = df.loc[df["relation"]==relation]

    df_relation = df_relation.reset_index(drop=True).drop(["relation", "no_echantillon"], axis=1)
    df_relation = pd.merge(df_relation.groupby("no_question").mean(numeric_only=True).round(4), df_relation.groupby("no_question").var(numeric_only=True).round(4), how="inner", left_on="no_question", right_on="no_question", suffixes=["_mean", "_var"])
    df_relation=df_relation.reset_index()
    df_relation["question"] = df_relation.apply(lambda row:  format_question(all_lf_questions[relation][int(row["no_question"])]), axis=1)
    df_relation.to_csv(f"summary_{relation}.csv")

    #plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots(figsize=(20,20))
    bars = ax.barh(df_relation["question"], df_relation["score0_mean"])
    for bar, label in zip(bars, df_relation["question"]):
        ax.text(-300000,
                bar.get_y()+ bar.get_height()/2,
                label, va="center", ha="right", color="black")
        
    ax.set_xlabel("Scores")

    ax.set_yticks([])
    plt.tight_layout()
    #plt.tight_layout()



    #df_relation.plot(x="question", y="score0_mean", kind="barh")
    #for i, question in enumerate(df_relation["question"]):
    # Format the label to be split into two lines (adjust as needed)
    #    label = f"Value: {question}More Info"
    
    # Place the label above each bar
    #    plt.text(i, 10, label, ha='center', va='bottom', fontsize=10)


    plt.savefig(f"{relation}_score.png")
    
    return df_relation



def create_df_by_k_shot(df, k_shot):
    summary = pd.merge( df.groupby(["relation", "no_question"]).mean().round(4), df.groupby(["relation", "no_question"]).var().round(4), how="inner", left_on=["relation", "no_question"], right_on=["relation", "no_question"], suffixes=["_mean", "_var"])
    
    summary = summary.drop(["no_echantillon_mean", "no_echantillon_var"], axis=1).reset_index()
    summary = summary[["relation", "no_question", f"score{k_shot}_mean", f"score{k_shot}_var"]]
    summary = summary.loc[summary.groupby("relation")[f"score{k_shot}_mean"].idxmax()]
    summary["question"] = summary.apply(lambda row:  format_question(all_lf_questions[row["relation"]][row["no_question"]]), axis=1)

    if k_shot > 0:
        summary["exemples"] = summary.apply(lambda row: get_examples(row["relation"]), axis=1)
        summary.columns = ["relation", "meilleure_question", "score", "variance_score", "question", "exemples"]
    else:
        summary.columns = ["relation", "meilleure_question", "score", "variance_score", "question"]


    
    summary.to_csv(f"bestQuestion_{k_shot}.csv")
    return summary

#create best question for all k-shot
def create_df_best_question(df):
    best =  pd.merge( df.groupby(["relation", "no_question"]).mean().round(4), df.groupby(["relation", "no_question"]).var().round(4), how="inner", left_on=["relation", "no_question"], right_on=["relation", "no_question"], suffixes=["_mean", "_var"])
    best = best.drop(["no_echantillon_mean", "no_echantillon_var"], axis=1).reset_index()
    
    best["score_mean"] = best.apply(lambda row: numpy.mean([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ]), axis=1)
    best["variation_of_scores"] = best.apply(lambda row: max([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ])- min([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ]), axis=1)
    best = best[["relation", "no_question", "score_mean", "variation_of_scores" ]]
    best = best.loc[best.groupby("relation")["score_mean"].idxmax()]
    best["question"] = best.apply(lambda row:  format_question(all_lf_questions[row["relation"]][row["no_question"]]), axis=1)
    best.columns = ["relation", "meilleure_question", "score", "difference_scores", "question"]
    print(best)
    histo = best.plot(x = "relation", y="score", title="Scores obtenus avec les meilleures questions pour chaque relation", kind="bar")
    
    
    #plt.rcParams["histo.figsize"] = (10, 5)
    #fig = histo.get_figure()
    plt.savefig('bestQuestionAll.pdf')

    best.to_csv("bestQuestionAll.csv")
    return best

def get_examples(rel):
    s = ''
    for w in k_exemples[rel]:
        s += w[0] + "->" + w[1] + ", "
    return s

def format_question(question):
    return question[0].replace('\\', '') + "x" + question[1]

def main():

    # dict0 = {"relation": "Anti", "no_question": 0, "no_echantillon":0, "score": 0.51}
    # dict1 = {"relation": "Anti", "no_question": 0, "no_echantillon":1, "score": 0.54}
    # dict2 = {"relation": "Anti", "no_question": 0, "no_echantillon":2, "score": 0.43}
    # dict3 = {"relation": "Anti", "no_question": 1, "no_echantillon":0, "score": 0.10}
    # dict4 = {"relation": "Anti", "no_question": 1, "no_echantillon":1, "score": 0.11}
    # dict5 = {"relation": "Anti", "no_question": 1, "no_echantillon":2, "score": 0.16}
    # dict6 = {"relation": "V_0", "no_question": 0, "no_echantillon":0, "score": 0.26}
    # dict7 = {"relation": "V_0", "no_question": 1, "no_echantillon":0, "score": 0.55}
    # df_0_shot = [dict0, dict1, dict2, dict3, dict4, dict5, dict6, dict7]

    # df0 = pd.DataFrame(df_0_shot)
    # var_test = 0
    # #print(df0)
    #
    # dict1 = {"relation": "Anti", "no_question": 0, "no_echantillon":1, f"score{var_test}": 0.51}
    # dict11 = {"relation": "Anti", "no_question": 0, "no_echantillon":1, "score1": 0.67}
    #
    # dict2 = {"relation": "Anti", "no_question": 0, "no_echantillon":2, "score0": 0.54, "score1": 0.50}
    # dict3 = {"relation": "Anti", "no_question": 1, "no_echantillon":0, "score0": 0.43, "score1": 0.20}
    # dict4 = {"relation": "Anti", "no_question": 1, "no_echantillon":1, "score0":0.10,  "score1": 0.45}
    # dict5 = {"relation": "Anti", "no_question": 1, "no_echantillon":2, "score0": 0.11,  "score1": 0.34}
    # dict6 = {"relation": "V_0", "no_question": 0, "no_echantillon":0, "score0":0.16, "score1": 0.56}
    # dict7 = {"relation": "V_0", "no_question": 1, "no_echantillon":0, "score0":0.26, "score1": 0.67}
    # df_1_shot = [dict1, dict11,  dict2, dict3, dict4, dict5, dict6, dict7]


    # dfall = pd.DataFrame(df_1_shot)
    # #print(dfall)
    #
    # df_anti0 = df0.loc[df0["relation"]=="Anti"].drop(["relation", "no_echantillon"], axis=1).reset_index(drop=True)
    #print("df")
    #print(df_anti0)

    # df_anti_all = dfall.loc[dfall["relation"]=="Anti"]
    # #print(df_anti_all)
    # df_anti_all = df_anti_all.reset_index(drop=True).drop(["relation", "no_echantillon"], axis=1)
    #df_anti = pd.merge(df0.loc[df0["relation"]=="Anti"], df1.loc[df1["relation"]=="Anti"], how="inner", left_on=["relation", "no_question", "no_echantillon"], right_on=["relation", "no_question", "no_echantillon"], suffixes=["0","1"]).reset_index(drop=True).drop(["relation", "no_echantillon"], axis=1)

    #df_anti = pd.concat([df0.loc[df0["relation"]=="Anti"], df1.loc[df1["relation"]=="Anti"]], axis=1, join="inner").drop_duplicates().reset_index(drop=True).drop(["relation", "no_echantillon"], axis=1)
    #df_anti.drop(["relation", "no_echantillon"], axis=1)

    #print(df_anti)

    # df_anti_all = pd.merge(df_anti_all.groupby("no_question").mean(numeric_only=True), df_anti_all.groupby("no_question").var(numeric_only=True), how="inner", left_on="no_question", right_on="no_question", suffixes=["_mean", "_var"])


    #print(df_anti_all)

    # k-shot
    # summary0 = dfall.groupby(["relation", "no_question"]).mean().drop("no_echantillon", axis=1).reset_index()
    # summary0 = summary0[["relation", "no_question", "score0"]]
    # #print(summary0)
    # summary0 = summary0.loc[summary0.groupby("relation")["score0"].idxmax()]
    # print("summary0")
    # #print(summary0)
    #
    # # changer pour df_complete.csv quand il sera généré
    testdf = pd.read_csv("df_complete.csv", index_col=0)
    testdf2 = testdf.round(2)
    for rel in all_lf_questions.keys():
        print(rel)
        create_df_by_relation(rel, df=testdf2)
    create_df_by_k_shot(testdf2, 0)
    create_df_by_k_shot(testdf2, 1)
    create_df_by_k_shot(testdf2, 3)
    create_df_by_k_shot(testdf2, 5)

    create_df_best_question(testdf2)




    # for k in [0,1,3,5]:
    #     process_samples('Anti', 30, 2, k)

    # process_samples('Anti', 30, 2)ds

    # for rel in all_lf_questions.keys():
    #     print(rel)
    #     process_samples(rel, 50, 3)

    # process_samples("A_2Perf", 50, 3, 0)

    # todo: à ameliorer (affichage, lecture du map)
    # open('./scores/fl_ranking', 'w', encoding="utf-8").close()
    # ranking_file = open('./scores/fl_ranking', 'a', encoding="utf-8")
    # fl_ranking_sorted = dict(sorted(fl_ranking.items(), key=lambda item: item[1], reverse=True))
    # for fl in fl_ranking_sorted.keys():
    #     ranking_file.write(str(fl) + '\n' + str(fl_ranking_sorted[fl][0]) + '|' + str(fl_ranking_sorted[fl][1][0]) + '\n')


if __name__ == "__main__":
    main()