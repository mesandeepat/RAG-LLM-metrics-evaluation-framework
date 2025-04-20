import os

import nltk
from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from langchain_openai import OpenAIEmbeddings

#os.environ["RAGAS_APP_TOKEN"] = ""
#os.environ["OPENAI_API_KEY"] = ""

nltk.data.path.append("/Users/sandeep/nltk_data")
llm = ChatOpenAI(model="gpt-4", temperature=0)
langchain_llm = LangchainLLMWrapper(llm)
embed = OpenAIEmbeddings()
loader = DirectoryLoader(
    path="/Users/sandeep/Downloads/LLM Evaluation_Resources/fs11",
    glob="**/*.docx",
    loader_cls=UnstructuredWordDocumentLoader
)
docs = loader.load()
generate_embeddings = LangchainEmbeddingsWrapper(embed)
generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=20)
print(dataset.to_list())
dataset.upload()



