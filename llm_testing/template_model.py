import json

from matplotlib.cbook import index_of
from matplotlib.colors import ListedColormap
from matplotlib.table import table
from pandas.io.sas.sas_constants import column_name_text_subheader_length
from tqdm import tqdm
import sys
import ollama
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Palette pour les graphiques
custom_palette = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]

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

# SOURCE: https://github.com/ollama/ollama/blob/main/examples/python-simplegenerate/client.py

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = 'llama3.2'

# global values for all samples file & n
file_name, example_file, example_lines, n = '', '', '', 0
all_fls_df = pd.DataFrame()
fl_ranking = {}

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

# Première façon de faire des requêtes au LLM: avec "post", requête par requête
# Génère la réponse du LLM
def generate(prompt, context, file, source_word, expected):
     print(prompt)

     r = requests.post('http://localhost:11434/api/generate',
                       json={
                           'model': model,
                           'prompt': "Tu es un modèle en français, ne me donne que des réponses d'un mot. " + prompt,
                           'context': context,
                       },
                       stream=True)
     r.raise_for_status()

     # file structure: source : expected : llm result
     file.write(f"{source_word} : {expected} : ")

     for i in r.iter_lines():
         body = json.loads(i)
         response_part = body.get('response', '')

         # the response streams one token at a time, print that as we receive it
         #print(response_part, end='', flush=True)
         file.write(response_part)

         if 'error' in body:
             raise Exception(body['error'])

         if body.get('done', False):
             file.write('\n')
             return body['context']
         
# Construit la question et le fichier de sortie
def process_question(relation_type: str, part1:str, part2:str, qIndex : int):
     # the context stores a conversation history, you can use this to make the model more context aware
     global context

     # Open a file to store the outputs
     fileName = f"outputs/{relation_type}_{qIndex}_out.csv"

     # Generate the prompt list to then give as arg to generate()
     examples = get_relation_examples(relation_type)

     # Clear file contents
     open("./"+fileName, 'w', encoding="utf-8").close()

     for e in tqdm(examples, desc="Processing examples"):
         source = e[0]
         expected = (" ").join(e[1:])
         question = part1 + source + part2
         context = generate(question, context, open(fileName, 'a', encoding="utf-8"), source, expected)
         sys.stdout.flush()

     return fileName

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
    examples = get_relation_examples(relation)
    scores_sublist = []

    # Clear score file
    open('./scores/' + relation, 'a', encoding="utf-8").close()

    for i in range (len(questions)):
        question = questions[i]

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
                    shots += ("Pour \""+k_exemples[relation][j][0]+"\", la réponse est \""+k_exemples[relation][j][1]+". \n")
                complete_question =  (str(question[0]) + source + str(question[1]) + "\n" + shots)
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

        output_file.close()

        score = success_rate(fileName)
        scores_sublist.append(score)

    return scores_sublist



# Process all the samples for a given relation
def process_samples(relation, sample_size, num_of_samples):
    # Use global values bc filename etc will be used later to get examples
    global file_name, example_file, example_lines, n, fl_ranking, all_fls_df

    chosen_relation = relation
    n = sample_size

    # Clear score file
    open('./scores/' + chosen_relation, 'w', encoding="utf-8").close()

    # Create list of dictionaries to transform in a Dataframe after
    list_of_dict = []

    # Iterate through each random sample
    for i in range(num_of_samples):
        scores_list = []
        file_name = f'./sample_sets/all_relations_{str(n)}_ex_{i}.tsv'
        examples_file = open(file_name, 'r', encoding="utf-8")

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

                dict1 = {"relation": relation, "no_question": l, "no_echantillon":i, f"score{k_shot[k]}" : round(scores_list[k][l], 2)}
                list_of_dict.append(dict1)

    df = pd.DataFrame(list_of_dict)
    concatenated_df = pd.concat([all_fls_df, df])
    all_fls_df = concatenated_df
    all_fls_df.to_csv("results/df_complete.csv")
    print(df)
    return all_fls_df

# Create dataframe with all the information for one relation
def create_df_by_relation(relation, df):
    df_relation = df.loc[df["relation"]==relation]

    df_relation = df_relation.reset_index(drop=True).drop(["relation", "no_echantillon"], axis=1)
    df_relation = pd.merge(df_relation.groupby("no_question").mean(numeric_only=True).round(4), df_relation.groupby("no_question").var(numeric_only=True).round(4), how="inner", left_on="no_question", right_on="no_question", suffixes=["_mean", "_var"])
    df_relation=df_relation.reset_index()
    df_relation["question"] = df_relation.apply(lambda row:  format_question(all_lf_questions[relation][int(row["no_question"])]), axis=1)
    df_relation.to_csv(f"results/summary_{relation}.csv")

    
    return df_relation


# Create dataframe with all the information for a given k-shot
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


    
    summary.to_csv(f"results/bestQuestion_{k_shot}.csv")
    return summary

#create a dataframe listing the best question for each relations, the score is a mean for all k-shot
def create_df_best_question(df):
    best =  pd.merge( df.groupby(["relation", "no_question"]).mean().round(4), df.groupby(["relation", "no_question"]).var().round(4), how="inner", left_on=["relation", "no_question"], right_on=["relation", "no_question"], suffixes=["_mean", "_var"])
    best = best.drop(["no_echantillon_mean", "no_echantillon_var"], axis=1).reset_index()
    
    best["score_mean"] = best.apply(lambda row: np.mean([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ]), axis=1)
    best["variation_of_scores"] = best.apply(lambda row: max([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ])- min([row["score0_mean"],row["score1_mean"], row["score3_mean"], row["score5_mean"] ]), axis=1)
    best = best[["relation", "no_question", "score_mean", "variation_of_scores" ]]
    best = best.loc[best.groupby("relation")["score_mean"].idxmax()]
    best["question"] = best.apply(lambda row:  format_question(all_lf_questions[row["relation"]][row["no_question"]]), axis=1)
    best.columns = ["relation", "meilleure_question", "score", "difference_scores", "question"]
    print(best)
    histo = best.plot(x = "relation", y="score", title="Scores obtenus avec les meilleures questions pour chaque relation", kind="bar")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()



    plt.savefig('results/bestQuestionAll.png')

    print(best)

    # make a graph showing the color of each k
    best.to_csv("results/bestQuestionAll.csv")
    return best

# Create dataframe and figure for the best question for each relations, showing score for different k 
def create_df_best_question_with_k(df):
    global custom_palette

    # get the whole df and the best score possible
    tout =  pd.merge( df.groupby(["relation", "no_question"]).mean().round(4), df.groupby(["relation", "no_question"]).var().round(4), how="inner", left_on=["relation", "no_question"], right_on=["relation", "no_question"], suffixes=["_mean", "_var"])

    results = {}
    for idx_tuple in tout.index:
        print(idx_tuple)
        if idx_tuple[0] not in results:
            results[idx_tuple[0]] = {}
        else:
            vals = results[idx_tuple[0]][idx_tuple[1]] = tout.loc[[idx_tuple], ["score0_mean", "score1_mean", "score3_mean", "score5_mean"]].values
            results[idx_tuple[0]][idx_tuple[1]] = vals[0].tolist()  # Extract the inner array and convert to list

    print(results)

    # get all k-shot values for each best question based on average score per k
    best_q_scores = {}
    best_qs = []

    for relation in results.keys():
        # for each relation, first calc the average and put in a list
        averages = {key: round(np.mean(values), 4) for key, values in results[relation].items()}
        print([relation])
        print(results[relation])
        # get the biggest avg to get the best question
        best_key = max(averages.items(), key=lambda x: x[1])[0]
        best_q_scores[relation] = results[relation][best_key]
        print(results[relation][best_key])
        best_qs.append((relation,
                        best_key,
                        round(np.var(results[relation][best_key]), 4),
                        (max(results[relation][best_key])),
                        results[relation][best_key].index(max(results[relation][best_key]))))

    # create the table to write in latex bc im lazy
    table_string = ''
    sep = ' & '
    for r in best_qs :
        table_string += (r[0] + sep + str(r[1]) + sep + str(r[4]) + sep + str(r[3]) + sep + str(r[2]) + "\\\\ \n")

    # mettre dans le bon format pour faciliter la création du graphique
    vals = {
        '0-shot' : [],
        '1-shot' : [],
        '3-shot': [],
        '5-shot' : []
    }
    for k in best_q_scores.keys():
        tab = best_q_scores[k]
        vals['0-shot'].append(tab[0])
        vals['1-shot'].append(tab[1])
        vals['3-shot'].append(tab[2])
        vals['5-shot'].append(tab[3])

    print(vals)

    color_map = {
        '0-shot': custom_palette[0],
        '1-shot': custom_palette[1],
        '3-shot': custom_palette[2],
        '5-shot': custom_palette[3]
    }

    # make the graph
    x = np.arange(len(best_q_scores))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    for k_shot, score in vals.items():
        offset = width * multiplier
        color = color_map.get(k_shot, 'gray')
        rects = ax.bar(x + offset, score, width, label=k_shot, color=color)
        multiplier += 1

    ax.set_xlabel("FL", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_title('Scores des meilleures questions pour chaque k-shot')
    ax.set_xticks(x + width, best_q_scores.keys())
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/bestQuestionAllKs.svg')
    plt.savefig('results/bestQuestionAllKs.pdf')
    plt.savefig('results/bestQuestionAllKs.png')

# Format examples in a more readable way
def get_examples(rel):
    s = ''
    for w in k_exemples[rel]:
        s += w[0] + "->" + w[1] + ", "
    return s

# Format questions in a more readable way
def format_question(question):
    return question[0].replace('\\', '') + "x" + question[1]

def main():

    df_complete = pd.read_csv("results/df_complete.csv", index_col=0)
    df_complete_rounded = df_complete.round(2)
    for rel in all_lf_questions.keys():
        print(rel)
        create_df_by_relation(rel, df=df_complete_rounded)

    create_df_best_question_with_k(df_complete_rounded)





if __name__ == "__main__":
    main()