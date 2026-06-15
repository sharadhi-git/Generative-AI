import ollama
client=ollama.Client()

model="deepseek-r1:1.5b"# give the name of open soruce code, asper in the ollama website
question="Hello, what is agentic AI"

response=client.generate(model=model,prompt=question)
print(response.response)