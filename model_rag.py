
from utils.etc import hit2docdict
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Modify
class ModelRAG():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def search(self, queries, qids, k=5):
        list_passages = []
        list_scores = []

        # fill here
        ######
        
        #### YOUR CODES; TODO 
        for query in queries:
            hits = self.retriever.search(query, k=k)
            
            passages = []
            scores = []
            
            for hit in hits:
                doc_dict = hit2docdict(hit)
                
                # Handling different possible keys in doc_dict
                title = doc_dict.get('title', doc_dict.get('Title', ''))
                text = doc_dict.get('text', doc_dict.get('contents', doc_dict.get('passage', '')))
                
                # Ensure we have some content
                if text.strip():
                    passages.append({
                        'title': title,
                        'text': text
                    })
                    scores.append(hit.score)
            
            list_passages.append(passages)
            list_scores.append(scores)
        ######

        return list_passages, list_scores

    # Modify
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # Get the relevant documents for each query
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        # fill here
        ######
        
        #### YOUR CODES; TODO 
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_text = ""
            for j, passage in enumerate(passages):
                title = passage.get('title', '')
                text = passage.get('text', '')
                if text.strip():
                    context_text += f"Passage {j+1}: {title} {text}\n"
            
            if not context_text.strip():
                context_text = "No relevant passages found.\n"
            
            input_text = f"Context:\n{context_text}Question: {query}\nAnswer:"
            list_input_text_without_answer.append(input_text)
        ######
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######
        
        #### YOUR CODES; TODO 
        list_input_text_without_answer = self.make_augmented_inputs_for_generate(queries, qids, k=k)
        
        # Tokenizing the inputs
        inputs = self.tokenizer(
            list_input_text_without_answer,
            padding=True,
            truncation=True,
            max_length=992,  
            return_tensors="pt"
        )
        
        supported_kwargs = {}
        if 'max_new_tokens' in kwargs:
            supported_kwargs['max_new_tokens'] = kwargs['max_new_tokens']
        if 'return_response_only' in kwargs:
            supported_kwargs['return_response_only'] = kwargs['return_response_only']
        kwargs = supported_kwargs
        ######

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs

class InputModelRAG(ModelRAG):
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_text = ""
            for j, passage in enumerate(passages):
                title = passage.get('title', '')
                text = passage.get('text', '')
                if text.strip():
                    context_text += f"Passage {j+1}: {title} {text}\n"
            
            if not context_text.strip():
                context_text = "No relevant passages found.\n"
            
            system_message = "You are a helpful assistant that answers questions based on the provided context. Give brief, direct answers."
            
            user_message = f"Context:\n{context_text}\nQuestion: {query}"
            
            input_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            list_input_text_without_answer.append(input_text)
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids, k=5, **kwargs):
        list_input_text_without_answer = self.make_augmented_inputs_for_generate(queries, qids, k=k)
        
        inputs = self.tokenizer(
            list_input_text_without_answer,
            padding=True,
            truncation=True,
            max_length=1920,  
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        

        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]
        return outputs
    
    def post_process_predictions(self, predictions):
        """Post-processing for Llama conversation format outputs"""
        import re
        
        processed = []
        
        for pred in predictions:
            cleaned = pred.replace('<|eot_id|>', '').replace('<|end_of_text|>', '').strip()
            cleaned = cleaned.replace('<|start_header_id|>', '').replace('<|end_header_id|>', '')
            
            cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
            
            if '\n' in cleaned:
                lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
                cleaned = lines[0] if lines else cleaned
            else:
                cleaned = cleaned.split('.')[0].strip()
            
            prefixes = ["Answer:", "The answer is:", "Based on the context:"]
            for prefix in prefixes:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            
            processed.append(cleaned if cleaned else "No answer")
        
        return processed

class ParsingModelRAG(ModelRAG):
    def __init__(self):
        super().__init__()
    
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_text = ""
            for j, passage in enumerate(passages):
                title = passage.get('title', '')
                text = passage.get('text', '')
                if text.strip():
                    context_text += f"Passage {j+1}: {title} {text}\n"
            
            # Fallback if no passages have content
            if not context_text.strip():
                context_text = "No relevant passages found.\n"
            
            input_text = f"""Context:
{context_text}
Question: {query}

Please provide your answer in the following JSON format:
{{"answer": "your brief direct answer here"}}

Answer:"""
            
            list_input_text_without_answer.append(input_text)
        
        return list_input_text_without_answer
    
    def post_process_predictions(self, predictions):
        """
        Post-process generated text to extract relevant parts for evaluation
        Handles lengthy outputs with additional explanations from instruction-tuned models
        """
        import re
        import json
        
        processed_predictions = []
        
        for pred in predictions:
            try:
                json_match = re.search(r'\{.*?\}', pred, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    parsed_json = json.loads(json_str)
                    

                    if 'answer' in parsed_json:
                        processed_predictions.append(str(parsed_json['answer']).strip())
                        continue
                    elif 'Answer' in parsed_json:
                        processed_predictions.append(str(parsed_json['Answer']).strip())
                        continue
            except (json.JSONDecodeError, AttributeError):
                pass
            
            cleaned_pred = pred.strip()
            
            # Remove common verbose prefixes that instruction-tuned models add
            verbose_prefixes = [
                "Based on the provided context,",
                "According to the context,",
                "Looking at the context,", 
                "From the context provided,",
                "The context indicates that",
                "Based on the information given,",
                "According to the passage,",
                "The answer is",
                "Answer:",
                "The correct answer is"
            ]
            
            for prefix in verbose_prefixes:
                if cleaned_pred.lower().startswith(prefix.lower()):
                    cleaned_pred = cleaned_pred[len(prefix):].strip()
                    break
            
            if '\n' in cleaned_pred:
                lines = [line.strip() for line in cleaned_pred.split('\n') if line.strip()]
                if lines:
                    cleaned_pred = lines[0]
            else:
                # Take first sentence (split by period)
                sentences = cleaned_pred.split('.')
                if sentences:
                    cleaned_pred = sentences[0].strip()
            
            # Remove leading articles and common filler words
            cleaned_pred = re.sub(r'^(the|a|an|is|was|are|were)\s+', '', cleaned_pred, flags=re.IGNORECASE).strip()
            
            processed_predictions.append(cleaned_pred)
        
        return processed_predictions
    

class COTModelRAG(ModelRAG):
    """
    Improved COT based on the user's working implementation
    Minimal changes to avoid breaking what's working
    """
    
    def __init__(self):
        super().__init__()
    
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        """Based on user's implementation with small improvements"""
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_parts = []
            # Only using top 3 passages to avoid overwhelming the model
            for j, passage in enumerate(passages[:3]):  
                title = passage.get('title', '').strip()
                text = passage.get('text', '').strip()
                
                if text:
                    if title and title.lower() not in text.lower()[:50]:
                        content = f"{title}: {text}"
                    else:
                        content = text
                    context_parts.append(content)
            
            if context_parts:
                context = "\n\n".join(context_parts)
                augmented_input = f"""Context: {context}

Question: {query}

Let's think step by step:
Answer:"""
            else:
                augmented_input = f"""Question: {query}

Let's think step by step:
Answer:"""
            
            list_input_text_without_answer.append(augmented_input)
        
        return list_input_text_without_answer

    def post_process_predictions(self, predictions):
        """
        Simple post-processing that works well with your prompt structure
        """
        import re
        
        processed = []
        
        for pred in predictions:
            if not pred or not pred.strip():
                processed.append("No answer")
                continue
                
            pred = pred.strip()
            
            answer_match = re.search(r"Answer:\s*([^\n]+)", pred, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).strip()
                processed.append(answer)
                continue

            lines = pred.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith(('1.', '2.', '3.', 'Step', 'Let')):
                    line = re.sub(r'^(The answer is|It is|This is|Based on)\s*:?\s*', '', line, flags=re.IGNORECASE)
                    if line and len(line) < 100:  # Reasonable length
                        processed.append(line)
                        break
            else:
                if len(pred) < 100:
                    processed.append(pred)
                else:
                    processed.append(pred[:50].strip())
        
        return processed


class SimpleCOTModelRAG(ModelRAG):
    """
    Very conservative improvement - almost identical to user's code
    """
    
    def __init__(self):
        super().__init__()
    
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        """Almost identical to user's implementation"""
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_parts = []
            for passage in passages[:3]:  
                text = passage.get('text', '').strip()
                if text:
                    context_parts.append(text)  
            
            if context_parts:
                context = "\n\n".join(context_parts)
                augmented_input = f"""Context: {context}

Question: {query}

Let's think step by step:
Answer:"""
            else:
                augmented_input = f"""Question: {query}

Let's think step by step:
Answer:"""
            
            list_input_text_without_answer.append(augmented_input)
        
        return list_input_text_without_answer

    def post_process_predictions(self, predictions):
        """
        Very simple post-processing
        """
        processed = []
        
        for pred in predictions:
            if not pred:
                processed.append("No answer")
                continue
                
            lines = [line.strip() for line in pred.split('\n') if line.strip()]
            if lines:
                first_line = lines[0]
                if first_line.lower().startswith('answer:'):
                    first_line = first_line[7:].strip()
                processed.append(first_line)
            else:
                processed.append("No answer")
        
        return processed
    

class InstructionAwareCompressionRAG(ModelRAG):
    """
    Enhanced RAG with Instruction-Aware Contextual Compression
    Fixed version with proper imports and error handling
    """
    
    def __init__(self, compression_ratio=0.7, min_sentences=2, max_sentences=8):
        super().__init__()
        self.compression_ratio = compression_ratio  
        self.min_sentences = min_sentences          
        self.max_sentences = max_sentences          
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  
        
    def _split_into_sentences(self, text):
        """Split text into sentences for compression analysis"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _calculate_instruction_relevance(self, sentences, instruction):
        """Calculate relevance score of each sentence to the instruction"""
        if not sentences or len(sentences) < 2:
            return np.ones(len(sentences))  
        

        all_texts = sentences + [instruction]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
            instruction_vector = tfidf_matrix[-1:]
            
            sentence_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(sentence_vectors, instruction_vector).flatten()
            
            return similarities
        except Exception as e:
            instruction_words = set(instruction.lower().split())
            scores = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(instruction_words.intersection(sentence_words))
                scores.append(overlap / max(len(instruction_words), 1))
            return np.array(scores)
    
    def _compress_context(self, context_text, instruction):
        """Compress context while preserving instruction-relevant information"""
        sentences = self._split_into_sentences(context_text)
        
        if len(sentences) <= self.min_sentences:
            return context_text  
        
        relevance_scores = self._calculate_instruction_relevance(sentences, instruction)
        
        if len(relevance_scores) == 0:
            return context_text
        
        target_sentences = max(
            self.min_sentences,
            min(
                self.max_sentences,
                int(len(sentences) * self.compression_ratio)
            )
        )
        

        if len(relevance_scores) <= target_sentences:
            top_indices = list(range(len(sentences)))
        else:
            top_indices = np.argsort(relevance_scores)[-target_sentences:]
        

        top_indices = sorted(top_indices)
        
        compressed_sentences = [sentences[i] for i in top_indices]

        compressed_text = '. '.join(compressed_sentences)
        if compressed_text and not compressed_text.endswith('.'):
            compressed_text += '.'
        
        return compressed_text
    
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        
        for i, query in enumerate(queries):
            passages = list_passages[i]
            
            context_text = ""
            for j, passage in enumerate(passages[:3]):  # Only top 3
                title = passage.get('title', '')
                text = passage.get('text', '')
                
                if text.strip():

                    compressed_text = self._compress_context(text, query)
                    if compressed_text.strip():
                        context_text += f"Passage {j+1}: {title} {compressed_text}\n"
            
            if not context_text.strip():
                context_text = "No relevant passages found.\n"
            
            input_text = f"Context:\n{context_text}Question: {query}\nAnswer:"
            list_input_text_without_answer.append(input_text)
        
        return list_input_text_without_answer
    
    def post_process_predictions(self, predictions):
        """Simple post-processing"""
        processed = []
        
        for pred in predictions:
            if not pred or not pred.strip():
                processed.append("No answer")
                continue
                
            lines = [line.strip() for line in pred.split('\n') if line.strip()]
            if lines:
                answer = lines[0]
                answer = re.sub(r'^(Answer:|The answer is:|Based on)\s*:?\s*', '', answer, flags=re.IGNORECASE)
                processed.append(answer.strip())
            else:
                processed.append("No answer")
        
        return processed