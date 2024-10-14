import json
import requests
from tqdm import tqdm
import sys

# SOURCE: https://github.com/ollama/ollama/blob/main/examples/python-simplegenerate/client.py

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = 'llama3.2' # TODO: update this for whatever model you wish to use

# Read file
file = open('./relations_100ex_tsv.tsv', 'r', encoding="utf-8")
lines = []
for line in file:
    l = line.replace('\n', '').split('\t')
    lines.append(l)

def get_relation_examples(rel_name):
    try:
        starting_index = next(i for i, t in enumerate(lines) if t == ['>>>', rel_name])
    except StopIteration:
        print(f"Relation '{rel_name}' not in examples.")
        return []

    examples = []
    for i in range(starting_index + 1, min(starting_index + 101, len(lines))):
        if lines[i]:
            examples.append(lines[i])
    return examples


def generate(prompt, context, file, source_word, expected):

    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                      },
                      stream=True)
    r.raise_for_status()

    file.write(f"{source_word} ; {expected} ; ")

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')

        # the response streams one token at a time, print that as we receive it
        #print(response_part, end='', flush=True)
        file.write(response_part)

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            file.write('\n')
            return body['context']

def process_question(relation_type: str, part1:str, part2:str):
    context = [] # the context stores a conversation history, you can use this to make the model more context aware

    # Open a file to store the outputs
    fileName = f"./{relation_type}_outputs.csv"
    # Generate the prompt list to then give as arg to generate()
    examples = get_relation_examples(relation_type)

    # Clear file contents
    open(fileName, 'w', encoding="utf-8").close()

    # for e in examples:
    #     source = e[0]
    #     question = f"Quel est le nom commun correspondant au verbe \"{source}\"? Donne un seul nom commun sans ponctuation"
    #     context = generate(question, context, open(fileName, 'a'), source)
        #print(context)

    for e in tqdm(examples, desc="Processing examples"):
        source = e[0]
        expected = (" ").join(e[1:])
        question = part1 + source + part2
        context = generate(question, context, open(fileName, 'a', encoding="utf-8"), source, expected)
        sys.stdout.flush()

def success_rate(filename):
    file = open('./'+filename, 'r', encoding="utf-8")
    lines = []
    for line in file:
        l = line.replace('\n', '').split(';')
        lines.append(l)

    print(lines)
    nb_success = 0
    total = 0

    for line in lines:
        if len(line)==3:
            source = str.lower(line[0])
            expected = str.lower(line[1])
            answer = str.lower(line[2]).strip()
            total+=1

            if answer in expected:
                nb_success += 1

    success_rate = nb_success/total
    return success_rate



def main():

    process_question('S_0', "Quel est le nom commun correspondant au verbe ou à l'adjectif \"", "\"? Donne un seul nom commun sans ponctuation" )
    process_question('Anti', "Quel est l'antonyme du mot \"", "\"? Donne un seul mot sans ponctuation.")

    print(success_rate("Anti_outputs.csv"))


    # while True:
    #     user_input = input("Enter a prompt: ")
    #     if not user_input:
    #         exit()
    #     print()
    #     context = generate(user_input, context)
    #     print()

if __name__ == "__main__":
    main()