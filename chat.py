import argparse

from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from text_generation import Client

PREPROMPT = "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, honest, sophisticated, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.\n"
PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.
{context}"""

END_7B = '\n<|prompter|>{query}<|endoftext|><|assistant|>'
END_40 = '\nUser: {query}\nFalcon:'

PARAMETERS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>", "</s>"],
}
CLIENT_7B = Client("http://")  # Fill this part
CLIENT_40B = Client("https://")  # Fill this part

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--top-k', type=str, default=32)
    parser.add_argument('--window-size', type=int, default=128)
    parser.add_argument('--step-size', type=int, default=100)
    return parser.parse_args()

def embed(fname, windows_size, step_size):
    text = extract_text(fname)
    text = " ".join(text.split())
    text_tokens = text.split()
    
    sentences = []
    for i in range(0, len(text_tokens), step_size):
        window = text_tokens[i:i+windows_size]
        if len(window) < windows_size:
            break
        
        sentences.append(window)
        
    paragraphs = [' '.join(s) for s in sentences]
    print(f'Number of paragraphs: {len(paragraphs)}')
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    model.max_seq_length = 512
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    embeddings = model.encode(paragraphs, show_progress_bar=True, convert_to_tensor=True)
    return model, cross_encoder, embeddings, paragraphs

def search(query, model, cross_encoder, embeddings, paragraps, top_k):
    query_embeddings = model.encode(query, convert_to_tensor=True)
    query_embeddings = query_embeddings.cuda()
    hits = util.semantic_search(query_embeddings, embeddings, top_k=top_k)[0]
    
    cross_input = [[query, paragraps[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_input)
    
    for idx in range(len(cross_scores)):
        hits[idx]['score'] *= cross_scores[idx]
    
    results = []
    hits = sorted(hits, key=lambda hit: hit['score'], reverse=True)
    for hit in hits:
        results.append({'score': hit['score'], 'text': paragraphs[hit['corpus_id']].replace('\n', ' ')})
    return results
    
    

if __name__ == '__main__':
    args = parse_args()
    model, cross_encoder, embeddings, paragraphs = embed(args.fname, args.window_size, args.step_size)
    print('Embedding done!')
    print(embeddings.shape)
    
    while True:
        print('\n')
        query = input('Enter query: ')
        results = search(query, model, cross_encoder, embeddings, paragraphs, top_k=args.top_k)
