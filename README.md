I built a RAG-based mini AI Query System (Chatbot) in the local environment using Python language, a simple HTML front-end with Hugging Face 'all-MiniLM-L6-v2' Model without using an API key. 
For this Chatbot, I used requirements like flask, pypdf2, numpy, faiss-cpu, and sentence-transformers.
I have used embeddings like Sentence Transformers and stored in vector data base like FAISS, using a PDF parser like PyPDF2. 
Some of the sample Queries to test the chatbot are: What is Machine learning? 
What is Deep learning? 
what are different layers in a transformer?
What are Neural Networks? 
What are different variations on transformer architecture?
Some of the limitations of this chatbot are: it generates responses within 400 tokens and in a limited manner within the uploaded document content. 
As it is a mini query system, which too built without an API key, it has a very limited scope to diversify to generate a response using the retrieved content. 
It can generate responses only based on the top 3 most relevant document chunks using vector similarity.
