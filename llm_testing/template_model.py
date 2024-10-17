import json
import requests
from tqdm import tqdm
import sys

# SOURCE: https://github.com/ollama/ollama/blob/main/examples/python-simplegenerate/client.py

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = 'llama3.2' # TODO: update this for whatever model you wish to use
context = []

# Read example tsv file
# Pour le moment, utiliser relations_50 car c'est plus rapide pour les tests
examples_file = open('./relations_50ex_tsv.tsv', 'r', encoding="utf-8")
example_lines = []

for line in examples_file:
    l = line.replace('\n', '').split('\t')
    example_lines.append(l)

# Get examples for a given relation from the chosen example file
def get_relation_examples(rel_name):
    try:
        starting_index = next(i for i, t in enumerate(example_lines) if t == ['>>>', rel_name])
    except StopIteration:
        print(f"Relation '{rel_name}' not in examples.")
        return []

    examples = []
    for i in range(starting_index + 1, min(starting_index + 101, len(example_lines))):
        if example_lines[i]:
            examples.append(example_lines[i])
    return examples

# Génère la réponse du LLM
def generate(prompt, context, file, source_word, expected):

    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                      },
                      stream=True)
    r.raise_for_status()

    # file structure: source ; expected ; llm result
    file.write(f"{source_word} ; {expected} ; ")

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
        l = line.replace('\n', '').split(';')
        lines.append(l)

    #print(lines)
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

    print(results)

    success_rate = nb_success/total
    return success_rate

# lf sera en fait une liste de questions pour chaque FL, afin de tester les différents types de verbalisation
# todo: figure out how to handle the 2 part questions

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


def main():

    # Automatiser les tests du modèle selon la fonction lexicale!!

    # tableaux de questions:

    all_lf_questions = {
        'Anti' : [
            ["Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation."]
                ],
        "S_0": [
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation"],
            ["Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"?"]
        ],
        "S_1": [
            # TODO
            []
        ]
    }

    #process_question('S_0', "Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation" )
    #process_question('Anti', "Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation.", 0)

    #print(success_rate("Anti_outputs.csv"))

    fl_choisie = "S_0"

    run_model_for(fl_choisie, all_lf_questions[fl_choisie])

    # todo: arranger l'affichage du progress bar

if __name__ == "__main__":
    main()