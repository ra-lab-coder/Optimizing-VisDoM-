import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from functools import partial
#from visdomrag_optimized_PL import VisDoMRAG, logger
from visdomrag_optimized import VisDoMRAG, logger
from pdf2image import convert_from_path
import os
import pandas as pd
import json
import time
import tqdm

class OptimizedVisDoMRAG(VisDoMRAG):
    def __init__(self, config):
        super().__init__(config)
        self.max_workers = config.get("max_workers", min(2, mp.cpu_count()))
        self.use_gpu_parallel = config.get("use_gpu_parallel", False)
        self.batch_size = config.get("batch_size", 4)
        
        # Pre-build indices once during initialization
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Pre-compute all indices once during initialization."""
        logger.info("Pre-computing indices for all queries...")
        
        # Build visual index once for all queries
        if not os.path.exists(self.vision_retrieval_file) or self.force_reindex:
            logger.info("Building visual index (one-time operation)...")
            self.build_visual_index()
        
        # Build text index once for all queries  
        if not os.path.exists(self.text_retrieval_file) or self.force_reindex:
            logger.info("Building text index (one-time operation)...")
            self.build_text_index()
        
        logger.info("✓ All indices pre-computed!")

    
    def retrieve_visual_contexts_optimized(self, query_id):
        """Optimized visual context retrieval - no index building."""
        try:
            # Load pre-computed retrieval results
            df_retrieval = pd.read_csv(self.vision_retrieval_file)
            
            # Filter for the current query
            query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
            if len(query_rows) == 0:
                logger.warning(f"No visual contexts found for query {query_id}")
                return []
            
            # Get top-k visual contexts
            top_k_rows = query_rows.nlargest(self.top_k, 'score')
            
            # Load the images from PDFs (this is the remaining bottleneck)
            pages = []
            pdf_dir = os.path.join(self.data_dir, "docs")
            
            for _, row in top_k_rows.iterrows():
                try:
                    document_id = row['document_id']
                    base_doc_id, page_number = document_id.rsplit('_', 1)
                    page_number = int(page_number)
                    
                    pdf_path = os.path.join(pdf_dir, f"{base_doc_id}.pdf")
                    if not os.path.exists(pdf_path):
                        continue
                    
                    # Convert only the specific page we need
                    pdf_images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
                    if pdf_images:
                        pages.append({
                            'image': pdf_images[0],
                            'document_id': document_id,
                            'page_number': page_number
                        })
                except Exception as e:
                    logger.error(f"Error loading PDF page for {row['document_id']}: {str(e)}")
            return pages
        
        except Exception as e:
            logger.error(f"Error retrieving visual contexts for query {query_id}: {str(e)}")
            return []


    def retrieve_textual_contexts_optimized(self, query_id):
        """Optimized textual context retrieval - no index building."""
        try:
            # Load pre-computed retrieval results directly
            df_retrieval = pd.read_csv(self.text_retrieval_file)
            
            # Filter for the current query
            query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
            if len(query_rows) == 0:
                return []
            
            # Get top-k textual contexts
            top_k_rows = query_rows.sort_values(by='rank', ascending=True).head(self.top_k)
            
            # Extract the contexts
            contexts = []
            for _, row in top_k_rows.iterrows():
                contexts.append({
                    'chunk': row['chunk'],
                    'chunk_pdf_name': row['chunk_pdf_name'] if 'chunk_pdf_name' in row else row.get('document_id', 'unknown'),
                    'pdf_page_number': row['pdf_page_number'] if 'pdf_page_number' in row else row.get('page_number', 0)
                })
           
            return contexts
        
        except Exception as e:
            logger.error(f"Error retrieving textual contexts for query {query_id}: {str(e)}")
            return []
    


    def process_query_optimized(self, query_id):
        """Optimized single query processing."""
        try:
            # Get query information
            query_row = self.df[self.df['q_id'] == query_id].iloc[0]
            question = query_row['question']
            
            try:
                answer = eval(query_row['answer'])
            except:
                answer = query_row['answer']
            
            # Define file paths for outputs
            visual_file = f"{self.output_dir}/{self.llm_model}_vision/response_{str(query_id).replace('/','$')}.json"
            textual_file = f"{self.output_dir}/{self.llm_model}_text/response_{str(query_id).replace('/','$')}.json"
            combined_file = f"{self.output_dir}/{self.llm_model}_visdmrag/response_{str(query_id).replace('/','$')}.json"
            
            # Skip if the combined file already exists
            if os.path.exists(combined_file):
                return True
            
            # Process visual and textual contexts in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                visual_future = executor.submit(self._process_visual_context, query_id, question, answer, visual_file)
                textual_future = executor.submit(self._process_textual_context, query_id, question, answer, textual_file)
                
                # Wait for both to complete
                visual_response_dict = visual_future.result()
                textual_response_dict = textual_future.result()
            
            # Skip if either response is missing
            if not visual_response_dict or not textual_response_dict:
                return False
            
            # Combine responses
            combined_sections = self.combine_responses(
                question, 
                visual_response_dict, 
                textual_response_dict,
                answer
            )
            
            # Create and save combined response
            combined_response = {
                "question": question,
                "answer": combined_sections.get("Final Answer", ""),
                "gt_answer": answer,
                "analysis": combined_sections.get("Analysis", ""),
                "conclusion": combined_sections.get("Conclusion", ""),
                "response1": visual_response_dict,
                "response2": textual_response_dict
            }
            
            with open(combined_file, 'w') as file:
                json.dump(combined_response, file, indent=4)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")
            return False
    
    def _process_visual_context(self, query_id, question, answer, visual_file):
        """Process visual context for a single query."""
        if os.path.exists(visual_file):
            with open(visual_file, 'r') as file:
                return json.load(file)
        
        visual_contexts = self.retrieve_visual_contexts_optimized(query_id)
        if not visual_contexts:
            return None
        
        visual_response = self.generate_visual_response(question, visual_contexts)
        visual_response_dict = self.extract_sections(visual_response)
        
        visual_response_dict.update({
            "question": question,
            "document": [ctx['document_id'] for ctx in visual_contexts],
            "gt_answer": answer,
            "pages": [ctx['page_number'] for ctx in visual_contexts]
        })
        
        with open(visual_file, 'w') as file:
            json.dump(visual_response_dict, file, indent=4)
        
        return visual_response_dict
    
    def _process_textual_context(self, query_id, question, answer, textual_file):
        """Process textual context for a single query."""
        if os.path.exists(textual_file):
            with open(textual_file, 'r') as file:
                return json.load(file)
        
        textual_contexts = self.retrieve_textual_contexts_optimized(query_id)
        if not textual_contexts:
            return None
        
        textual_response = self.generate_textual_response(question, textual_contexts)
        textual_response_dict = self.extract_sections(textual_response)
        
        textual_response_dict.update({
            "question": question,
            "document": [ctx['chunk_pdf_name'] for ctx in textual_contexts],
            "gt_answer": answer,
            "pages": [ctx['pdf_page_number'] for ctx in textual_contexts],
            "chunks": "\n".join([ctx['chunk'] for ctx in textual_contexts])
        })
        
        with open(textual_file, 'w') as file:
            json.dump(textual_response_dict, file, indent=4)
        
        return textual_response_dict
    
    def run_parallel(self):
        """Run the VisDoMRAG pipeline with parallel processing."""
        logger.info("Starting optimized VisDoMRAG pipeline with parallel processing")
        
        query_ids = self.df['q_id'].unique()
        successful_queries = 0
        failed_queries = 0
        
        # Process queries in parallel batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(self.process_query_optimized, query_id): query_id 
                for query_id in query_ids
            }
            
            # Process results as they complete
            for future in tqdm.tqdm(as_completed(future_to_query), total=len(query_ids), desc="Processing queries"):
                query_id = future_to_query[future]
                try:
                    success = future.result(timeout=300)  # 5 minute timeout per query
                    if success:
                        successful_queries += 1
                        logger.info(f"✓ Successfully processed query {query_id}")
                    else:
                        failed_queries += 1
                        logger.warning(f"✗ Failed to process query {query_id}")
                        
                except Exception as e:
                    failed_queries += 1
                    logger.error(f"✗ Error processing query {query_id}: {str(e)}")
        
        logger.info(f"Pipeline completed: {successful_queries} successful, {failed_queries} failed")
    
    def run_gpu_parallel(self):
        """Run with GPU parallelization for API-based models."""
        logger.info("Starting GPU-parallel VisDoMRAG pipeline")
        
        query_ids = self.df['q_id'].unique()
        
        # Process in smaller batches to avoid API rate limits
        batch_size = self.batch_size
        
        for i in tqdm.tqdm(range(0, len(query_ids), batch_size), desc="Processing batches"):
            batch_queries = query_ids[i:i + batch_size]
            
            # Process batch with threading (good for I/O bound API calls)
            with ThreadPoolExecutor(max_workers=min(len(batch_queries), 4)) as executor:
                futures = [
                    executor.submit(self.process_query_optimized, query_id)
                    for query_id in batch_queries
                ]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")
            
            # Small delay between batches to respect API limits
            if i + batch_size < len(query_ids):
                time.sleep(2)

if __name__ == "__main__":
# Example usage of optimized VisDoMRAG

# Configuration with optimization settings
    config = {
    "data_dir": "data/feta_tab_optimised",
    "output_dir": "results/feta_tab_results_colqwen_bge_llama_optimised", 
    "llm_model": "meta/meta-llama-3-70b-instruct",
"llm_provider": "llama",
    "vision_retriever": "colqwen",
    "text_retriever": "bge",
    "top_k": 5,
    
    # Optimization settings
    "max_workers": 2,  # Number of parallel processes
    "use_gpu_parallel": False,  # Use GPU parallelization
    "batch_size": 4,  # Batch size for API calls
    "force_reindex": False,  # Use pre-computed indices
    "force_cpu_vision": True,

    
    "api_keys": {
        "gemini": " ",
         "mistral":"",
        "replicate": "",
    }
}

visdmrag = OptimizedVisDoMRAG(config)
visdmrag.run_parallel()
