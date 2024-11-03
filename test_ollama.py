import ollama 

modelfile='''
FROM llama3.2
SYSTEM You are mario from super mario bros.
'''

# ollama.create(model='example1', modelfile=modelfile)

#print(ollama.list())

response = ollama.chat(model='question-llama2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])


#print(ollama.chat(model="mario",messages=[{'content':'hello'}]))