import json
#import requests
from tqdm import tqdm
import sys
import ollama


all_lf_questions = {
        'Anti' : [
            ["Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le contraire du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est l'opposé du mot \"", "\"? Donne un seul mot sans ponctuation."]
                ],
        "S_0": [
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."],
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation."]
        ],
        "Syn_⊂" : [
            ["Quel est le synomyme avec un sens plus large du mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Donne un seul mot qui englobe le sens du mot \"", "\"."],
            ["Donne un mot plus général pour signifier \"", "\"."]
        ], 
        "Syn" : [
            ["Quel est le synonyme du mot \"", "\"? Donne un seul mot sans ponctuation."]
        ],
        "A_0" : [
            ["Quel est l'adjectif correspondant au mot \"", "\"? Donne un seul adjectif sans ponctuation."]
        ],
        "A_2Perf" : [
            ["Quel est l'adjectif correspondant à l'aboutissement de \"", "\"? Donne un seul adjectif sans ponctuation."],
            ["Quel est l'adjectif correspondant à l'aboutissement de \"", 
             "\"? Donne un seul adjectif sans ponctuation. Voici un exemple: l'adjectif correspondant à l'aboutissement de \"cuire\" est \"cuit\"."]
        ],
        "V_0" : [
            ["Quel est le verbe correspondant au mot \"", "\"? Donne un seul verbe sans ponctuation."],
        ],
        "Syn_⊃^sex" : [
            ["Quel est le mot féminin correspondant au mot \"", "\"? Donne un seul mot sans ponctuation."],
            ["Quel est le correspondant féminin du mot \"", "\"? Donne un seul mot sans ponctuation."]
        ],
        "Adv_0" : [
            ["Quel est l'adverbe correspondant au mot \"", "\"? Donne un seul adverbe sans ponctuation."]
        ],
        "S_instr" : [
            ["Quel est l'instrument typiquement utilisé pour faire l'action liée au mot \"", "\"? Donne un seul nom commun sans ponctuation."]
        ],
        "Magn" : [
            ["Quel est le mot utilisé avec le mot \"", "\" qui amplifie son sens? Donne un seul mot sans ponctuation."],
            ["Quel est le mot utilisé avec le mot \"", "\" qui intesifie son sens? Donne un seul mot sans ponctuation."],
        ],
        "Redun" : [
            ["Donne un mot qui est utilisé comme modificateur redondant du mot \"", "\". Donne un seul mot sans ponctuation."],
            ["Donne un mot dont le sens est inclu dans celui du mot \"", "\". Donne un seul mot sans ponctuation."]
        ],
        "S_loc": [
            ["Quel est un nom qui décrit la localisation de \"", "\"? Donne un seul nom sans ponctuation."]
        ]
    
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

# Read example tsv file
# Pour le moment, utiliser relations_50 car c'est plus rapide pour les tests

file_name = './sample_sets/all_relations_100_ex_0.tsv'

examples_file = open(file_name, 'r', encoding="utf-8")
example_lines = []

for line in examples_file:
    l = line.replace('\n', '').split('\t')
    example_lines.append(l)

# Get examples for a given relation from the chosen example file
def get_relation_examples(rel_name):
    global example_lines, file_name

    # todo: gérer si 50 ou 100
    num = int(file_name[28]+file_name[29]+(file_name[30] if file_name[30]=='0' else ''))


    try:
        starting_index = next(i for i, t in enumerate(example_lines) if t == ['>>>', rel_name])
    except StopIteration:
        print(f"Relation '{rel_name}' not in examples.")
        return []

    examples = []
    for i in range(starting_index + 1, min(starting_index + num + 1, len(example_lines))):
        if example_lines[i]:
            examples.append(example_lines[i])
    return examples

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

# lf sera en fait une liste de questions pour chaque FL, afin de tester les différents types de verbalisation
def run_model_for(lf: str, lf_questions : list):
    # open the SR file (empty)
    sr_filename = f"./results/sr_{lf}"
    open(sr_filename, 'w', encoding="utf-8").close()
    sr_file = open(sr_filename, 'a', encoding="utf-8")

    # q est séparée en 2 parties, "quel est xxx" et "donne xxx"
    for q in lf_questions:
        # for each question, run the model & add SR to the SR file
        fl_file = process_question(lf, q[0], q[1], lf_questions.index(q))
        sr_file.write(
            str(lf_questions.index(q)) + " " +
            str(q[0]) + str(q[1]) + ", sr = " +
            str(success_rate(fl_file)) + "\n"
        )


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


# Rouler le modele
def run_model(relation):
    questions = all_lf_questions[relation]

    for i in range (len(questions)):
        question = questions[i]

        # Open a file to store the outputs
        fileName = f"outputs/{relation}_{i}_out.csv"

        # Clear file contents
        open("./"+fileName, 'w', encoding="utf-8").close()
        file = open("./"+fileName, 'a', encoding="utf-8")

        examples = get_relation_examples(relation)

        for w in tqdm(examples, desc="Processing examples"):
            source = w[0]
            complete_question =  (str(question[0]) + source + str(question[1]))
            response = ollama.chat(model='relation_general_model', messages=[
                {
                'role': 'user',
                'content': complete_question,
                },
            ])

            expected = (" ").join(w[1:])

            # file structure: source : expected : llm result
            file.write(f"{source} : {expected} : {response['message']['content']}")


            print(complete_question)
            print((f"{source} : {expected} : {response['message']['content']}"))


        file.close()

# Mettre juste les sources (pour le moment) des exemples dans un txt ou les valeurs sont separees par des virgules
# Pour les donner à chainforge
def examples_in_list(examples):
    f = open("examples.txt", "a")
    liste = []
    for w in examples:
        f.write(w[0] + ',')
        liste.append(w[0])
    f.close()
    return liste

# combiner les prompts en un gros prompt pour simuler le traitement en batch et eviter de re-prompt a chaque fois
def combined_prompts(question, words):
    global context

    questions = ""
    questions_list = []
    print(words)
    for w in words:
        questions += (str(question[0]) + w[0] + str(question[1])) + '\n'
        questions_list.append((str(question[0]) + w[0] + str(question[1])))

    # Open a file to store the outputs
    fileName = 'test.txt'

    # Generate the prompt list to then give as arg to generate()
    # Clear file contents
    open("./" + fileName, 'w', encoding="utf-8").close()

    print(questions)

    #context = generate(questions, context, open(fileName, 'a', encoding="utf-8"), '', '')

    sys.stdout.flush()

    return questions_list


def main():


    run_model("Syn")



if __name__ == "__main__":
    main()