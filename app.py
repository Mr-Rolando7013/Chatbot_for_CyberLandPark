import gradio as gr
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("CAF.pdf")
docs = loader.load()

# Join all pages into one string
caf_text = "\n".join([d.page_content for d in docs])

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()